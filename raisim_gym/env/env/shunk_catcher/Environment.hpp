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

/* Convention
*   action space = force (continuous)                        n = 1, index =0
*
*   observation space = [ x (cart position)                  n =  1, index=0
                          theta (tilted angle),              n =  1, index=1
                          x_dot (cart linear velocity),      n =  1, index=2
*                         theta_dot (tilted angle velocity,  n =  1, index=3] total 4
*/


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
            hand->setBasePos_e(Eigen::Vector3d(0, 0, 0.3));

            Eigen::Matrix3d rotation;
//            rotation << 0,0,1, 0,1,0, -1,0,0;
//            rotation << 1,0,0, 0,0,-1, 0,1,0, 0,0,0; //rot x 90
            rotation << 1,0,0, 0,-0.7,-0.7, 0,0.7,-0.7, 0,0,0; //rot x 45
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
            hand_gc_.setZero(handGcDim_); hand_gc_init_.setZero(handGcDim_);
            hand_gv_.setZero(handGvDim_); hand_gv_init_.setZero(handGvDim_);

            hand_prev_gc_= hand_gc_init_.tail(actionDim_);

//            RSINFO(hand_->getGeneralizedCoordinate());
//            gc_init_ << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ,0.1 ,0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

            ///Ball
            ball_ = world_->addSphere(0.03, 0.03);
            ball_gc_init = {0,-0.07,0.5};
            ball_->setPosition(ball_gc_init);
            ball_->setVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));
            auto ground = world_->addGround();
            world_->setERP(0, 0);

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 1;
            actionDim_ = 20;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obMean_.setZero(obDim_);
            obStd_.setZero(obDim_);

            /// action & observation scaling
            actionMean_ = hand_gc_init_.tail(actionDim_);
            actionStd_.setConstant(0.1);

            obMean_.setZero();
            obStd_ << ball_gc_init[2];

            //Reward coefficients
            forceRewardCoeff_ = cfg["forceRewardCoeff"].as<double>();
            actionScaler_ = cfg["actionScaler"].as<double>();

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
//                cartpoleVisual_ = vis->createGraphicalObject(cartpole_, "Cartpole");
                ballVisual_ = vis->createGraphicalObject(ball_, "ball", "default");
                handVisual_ = vis->createGraphicalObject(hand_, "hand");


                vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
                desired_fps_ = 50.;
                vis->setDesiredFPS(desired_fps_);
//                vis->select(cartpoleVisual_->at(0), false);
                vis->select(handVisual_->at(0), false);

                vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-1.0), Ogre::Radian(-1.0), 1);
            }
        }

        double fRand(double fMin, double fMax)
        {
            double f = (double)rand() / RAND_MAX;
            return fMin + f * (fMax - fMin);
        }

        ~ENVIRONMENT() final = default;

        void init() final {}

        void reset() final {

            auto ball_h = ball_gc_init;

            ball_h[2] += fRand(-0.1, 3);

            ball_->setPosition(ball_h);
            ball_->setVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));

            hand_prev_gc_= hand_gc_init_.tail(actionDim_);

            hand_->setState(hand_gc_init_, hand_gv_init_);
            updateObservation();
            if (visualizable_)
                gui::rewardLogger.clean();
//            RSINFO("er");

        }

        float step(const Eigen::Ref <EigenVec> &action) final {
//            RSINFO("step");

            /// action scaling
            actionScaled_ =  hand_prev_gc_+action.cast<double>()*actionScaler_;
            Eigen::VectorXd jointVelocityTarget(handGvDim_);
            jointVelocityTarget.setZero();
            hand_->setPdTarget(actionScaled_, jointVelocityTarget);
            hand_prev_gc_ = actionScaled_;

            /// add random ball velocity
            raisim::Vec<3>  hand_pos{}, ball_pos{}, ball_speed{}, rand_component{};
            ball_->getPosition(0, ball_pos);
            hand_->getPosition(7, hand_pos);
            ball_->getVelocity(0, ball_speed);

            if (ball_pos[2]<0.4){
                rand_component = {fRand(-0.2,0.2),fRand(-0.2,0.2),fRand(-0.2,0.2)};
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
            ball_pos-=hand_pos;
            reward_ = forceRewardCoeff_ * 1/ball_pos.squaredNorm();

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
//            RSINFO("uo");

            obDouble_.setZero(obDim_);
            obScaled_.setZero(obDim_);
            obDouble_ << ball_->getPosition()[2];
            obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);
//            RSINFO("euo");

        }

        void observe(Eigen::Ref <EigenVec> ob) final {
            /// convert it to float
//            RSINFO("ob");
            ob = obScaled_.cast<float>();
//            RSINFO("after ob");

        }

        bool isTerminalState(float &terminalReward) final {
//            RSINFO("ist");

            terminalReward = float(terminalRewardCoeff_);
            //If the angle of pole is greater than +-50 degs or the cart position is greater than +-2m,
            //treat them as terminal conditions
            if (ball_->getPosition()[2] < 0.15) {
//                RSINFO(ball_->getPosition()[2]);

                return true;
            }
            terminalReward = 0.f;
//            RSINFO("not t");

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

//        raisim::ArticulatedSystem *cartpole_;
//        std::vector <GraphicObject> *cartpoleVisual_;

        // Ball
        raisim::Sphere *ball_;
        std::vector <GraphicObject> *ballVisual_;
        raisim::Vec<3> ball_gc_pos,ball_gc_init;
        //Hand
        raisim::ArticulatedSystem *hand_;
        std::vector <GraphicObject> *handVisual_;
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

