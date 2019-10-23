//
// Created by Jemin on 3/27/19.
// Modified by Inkyu on 22/July/2019 (based on Laikago Environment)
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include <stdlib.h>
#include <cstdint>
#include <set>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
#include "visualizer/raisimBasicImguiPanel.hpp"

using namespace std;
#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

        raisim::ArticulatedSystem *load_hand() {
            static int cointer = 0;
            cointer++;

            auto path = resourceDir_ +"/urdf/svh.urdf";
            cout << path << endl;

            auto hand = world_->addArticulatedSystem(path); // initialized to zero angles and identity orientation. Use setState() for a specific initial condition  jointPgain.setZero();
            hand->updateMassInfo();

            // world.setTimeStep(0.00006);
            int runt_dof = hand->getDOF();
            Eigen::VectorXd jointPgain(runt_dof), jointDgain(runt_dof);
            int Pgain = 200;
            int Dgain = 80;
            jointPgain.tail(runt_dof).setConstant(Pgain);
            jointDgain.tail(runt_dof).setConstant(Dgain);

            Eigen::VectorXd limits_max(runt_dof), limits_min(runt_dof);
            float limit = 500.0;

            limits_max.tail(runt_dof).setConstant(limit);
            limits_min.tail(runt_dof).setConstant(-limit);

            hand->setName("hand" + std::to_string(cointer));

            Eigen::VectorXd start_pose(runt_dof + 1);
            start_pose.tail(runt_dof + 1).setConstant(0.1);
            hand->setBasePos_e(initPosition);

            Eigen::Matrix3d rotation;
//            rotation << 0,0,1, 0,1,0, -1,0,0;
//            rotation << 1,0,0, 0,0,-1, 0,1,0, 0,0,0; //rot x 90
            rotation << 1,0,0, 0,-0.7,-0.7, 0,0.7,-0.7; //rot x 45
            hand->setBaseOrientation_e(rotation);

            hand->setGeneralizedCoordinate(start_pose); //

            hand->setActuationLimits(limits_max, limits_min);
            hand->setControlMode(
                    raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE); //raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE

            hand->setPdGains(jointPgain, jointDgain);

//            vis->createGraphicalObject(hand, );
            return hand;
        }

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const YAML::Node &cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), distribution_(0.0, 0.2), visualizable_(visualizable) {

            /// add objects

            /// Hand
            hand_ = load_hand();
            handGcDim_ = hand_->getGeneralizedCoordinateDim();
//            RSINFO(gcDim_);
            handGvDim_ = hand_->getDOF();
            nJointsHand_ = 9;
            hand_gc_.setZero(handGcDim_); hand_gc_init_.setZero(handGcDim_); hand_gc_init_.tail(handGcDim_).setConstant(0.1);
            hand_gv_.setZero(handGvDim_); hand_gv_init_.setZero(handGvDim_);

            hand_prev_gc_= hand_gc_init_.tail(handGcDim_);

            ///Ball
            ball_ = world_->addSphere(0.03, 0.03);
            ball_gc_init = raisim::Vec<3>{0,-0.07,0.2};
            ball_gc_init[0] += initPosition[0];            ball_gc_init[1] += initPosition[1];            ball_gc_init[2] += initPosition[2];

            ball_->setPosition(ball_gc_init);
            ball_->setVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));
            auto ground = world_->addGround();
            world_->setERP(0, 0);

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 7;//ball(x,y,z), handx - ball_x,hand_y-ball_y, hand_z-ball_z), dist, d(dist)
            actionDim_ = handGcDim_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obMean_.setZero(obDim_);
            obStd_.setZero(obDim_);

            /// action & observation scaling
            actionMean_ = hand_gc_init_.tail(actionDim_);
            actionStd_.setConstant(0.1);

            obMean_.setZero();
            obStd_ << 1,1,ball_gc_init[2], 1,1,1, ball_gc_init[2];

            //Reward coefficients
            forceRewardCoeff_ = cfg["forceRewardCoeff"].as<double>();
            actionScaler_ = cfg["actionScaler"].as<double>();
            minBallHeight_ = cfg["minBallHeight"].as<double>();

            auto speedRand = cfg["lowBallSpeedRand"];
            speedRandomMin_={speedRand[0].as<double>(),speedRand[2].as<double>(),speedRand[4].as<double>()};
            speedRandomMax_={speedRand[1].as<double>(),speedRand[3].as<double>(),speedRand[5].as<double>()};

            auto coordRand = cfg["ballSpawnCoordRand"];
            coordRandom_={coordRand[0].as<double>(),coordRand[1].as<double>(),coordRand[2].as<double>()};

            auto randParallelShift = cfg["randParallelShift"];
            randParallelShiftMin_={randParallelShift[0].as<double>(),randParallelShift[2].as<double>(),randParallelShift[4].as<double>()};
            randParallelShiftMax_={randParallelShift[1].as<double>(),randParallelShift[3].as<double>(),randParallelShift[5].as<double>()};

            gui::rewardLogger.init({"reward", "forceReward"});
            reward_ = 0;

            /// visualize if it is the first environment
            if (visualizable_) {
                auto vis = raisim::OgreVis::get();

                /// these method must be called before initApp
                vis->setWorld(world_.get());
                vis->setWindowSize(1280, 720);
                vis->setImguiSetupCallback(imguiSetupCallback);
                vis->setImguiRenderCallback(imguiRenderCallBack);
                vis->setKeyboardCallback(raisimKeyboardCallback);
                vis->setSetUpCallback(setupCallback);
                vis->setAntiAliasing(2);

                /// starts visualizer thread
                vis->initApp();
                ballVisual_ = vis->createGraphicalObject(ball_, "ball", "default");
                handVisual_ = vis->createGraphicalObject(hand_, "hand");


                vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
                desired_fps_ = 50.;
                vis->setDesiredFPS(desired_fps_);
                vis->select(handVisual_->at(0), false);

                vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-1.0), Ogre::Radian(-1.0), 1);
            }
        }

        double fRand(double fMin, double fMax){
            double f = (double)rand() / RAND_MAX;
            return fMin + f * (fMax - fMin);
        }

        raisim::Vec<3> vec3Rand(raisim::Vec<3> min, raisim::Vec<3> max){
            return raisim::Vec<3>{fRand(min[0],max[0]),fRand(min[1],max[1]),fRand(min[2],max[2])};
        }

        ~ENVIRONMENT() final = default;

        void init() final {}

        void reset() final {
            auto randShift = vec3Rand(randParallelShiftMin_, randParallelShiftMax_);

            auto ball_coord = ball_gc_init;
            ball_coord += vec3Rand(raisim::Vec<3>{0,0,0},coordRandom_);
            ball_coord += randShift;
            ball_->setPosition(ball_coord);
            ball_->setVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));

            hand_prev_gc_= hand_gc_init_.tail(handGcDim_);
            hand_->setGeneralizedCoordinate(hand_gc_init_);
            auto new_hand_pos = initPosition;
            new_hand_pos[0]+= randShift[0];new_hand_pos[1]+= randShift[1];new_hand_pos[2]+= randShift[2];
            hand_->setBasePos_e(new_hand_pos);
            updateObservation();
            if (visualizable_)
                gui::rewardLogger.clean();
//            RSINFO("er");

        }

        float getDistanceToBall(){
            int hand_joint = 6;
            raisim::Vec<3>  hand_pos{}, finger_pos{}, ball_pos{};
            ball_->getPosition(0, ball_pos);
            hand_->getPosition(hand_joint, hand_pos);
//            RSINFO(ball_pos);
//            RSINFO(hand_pos);

//            hand_->getPosition(hand_joint, finger_pos);
//            RSINFO(finger_pos);
//
//            ///pose near mid of palm
//            hand_pos[2] = hand_pos[2]+(hand_pos[2]-finger_pos[2])/2;

            ball_pos-=hand_pos;
            auto dist = ball_pos.squaredNorm();
//            RSINFO(dist);
//            RSINFO("");

            return dist;
        }

        float step(const Eigen::Ref <EigenVec> &action) final {
//            RSINFO("step");

            /// action scaling
            actionScaled_ =  hand_prev_gc_+action.cast<double>()*actionScaler_;
//            RSINFO(actionScaled_);
            Eigen::VectorXd jointVelocityTarget(handGvDim_);
            jointVelocityTarget.setZero();
            hand_->setPdTarget(actionScaled_, jointVelocityTarget);
            hand_prev_gc_ = actionScaled_;

            /// add random ball velocity
            raisim::Vec<3> ball_speed{}, rand_component{};
            ball_->getVelocity(0, ball_speed);
            float distToBall = getDistanceToBall();

            if (distToBall < 0.1){
                rand_component = {vec3Rand(speedRandomMin_,speedRandomMax_)};
                ball_speed += rand_component;
                ball_->setVelocity(ball_speed, {0,0,0});
            }


            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
            auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

            for (int i = 0; i < loopCount; i++) {
                world_->integrate();

                if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
                    raisim::OgreVis::get()->renderOneFrame();

                visualizationCounter_++;
            }
            updateObservation();

            /// check distance from some point on hand and calc rawrd

            reward_ = forceRewardCoeff_ * 1/distToBall;

            if (visualizeThisStep_) {
                gui::rewardLogger.log("forceReward", forceReward_);
                gui::rewardLogger.log("reward", reward_);
            }
//            RSINFO("step");
            return reward_;

        }

        void updateExtraInfo() final {
//            RSINFO("ei");

            extraInfo_["reward"] = reward_;
//            RSINFO("eei");

        }

        void updateObservation() {
            raisim::Vec<3> ball_pos{}, hand_pos{}, dpos{}, vel{};
            ball_->getPosition(0, ball_pos);
            hand_->getPosition(6, hand_pos);
            dpos = hand_pos;
            dpos -= ball_pos;
            obDouble_.setZero(obDim_);
            obScaled_.setZero(obDim_);
            obDouble_ << ball_pos[0],ball_pos[1],ball_pos[2], dpos[0], dpos[1], dpos[2], getDistanceToBall();
            obScaled_ = obDouble_;//(obDouble_ - obMean_).cwiseQuotient(obStd_);

        }

        void observe(Eigen::Ref <EigenVec> ob) final {
            ob = obScaled_.cast<float>();
        }

        bool isTerminalState(float &terminalReward) final {
//            RSINFO("ist");

            terminalReward = float(terminalRewardCoeff_);
            if (ball_->getPosition()[2] < minBallHeight_) {
//                RSINFO(ball_->getPosition()[2]);
                return true;
            }
            terminalReward = 0.f;

            return false;
        }

        void setSeed(int seed) final {
//            RSINFO("sseed");

            std::srand(seed);
//            RSINFO("e sseed");

        }

        void close() final {
        }

    private:
        double reward_;

//        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;

        std::normal_distribution<double> distribution_;

        Eigen::Vector3d initPosition{0,0,0.3};
        // Ball
        raisim::Sphere *ball_;
        std::vector <GraphicObject> *ballVisual_;
        raisim::Vec<3> ball_gc_pos,ball_gc_init;
        raisim::Vec<3> speedRandomMin_,speedRandomMax_, coordRandom_;
        float minBallHeight_=0;
        //Hand
        raisim::ArticulatedSystem *hand_;
        std::vector <GraphicObject> *handVisual_;
        raisim::Vec<3> randParallelShiftMin_,randParallelShiftMax_;
        Eigen::VectorXd hand_gc_init_, hand_gv_init_, hand_gc_,hand_prev_gc_, hand_gv_;
        int handGvDim_,handGcDim_,nJointsHand_;

//        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, actionScaled_, torque_;
        Eigen::VectorXd actionScaled_;
        double terminalRewardCoeff_ = -10.;
        double forceRewardCoeff_ = 0., forceReward_ = 0., actionScaler_ = 1.;
        double desired_fps_ = 60.;
        int visualizationCounter_ = 0;
        Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
        Eigen::VectorXd obDouble_, obScaled_;
    };

}

