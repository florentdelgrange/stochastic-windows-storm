//
// Created by Florent Delgrange on 2019-08-06.
//

#include <storm/settings/modules/MinMaxEquationSolverSettings.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentClassifier.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionUnfolding.h>
#include "gtest/gtest.h"
#include "storm-parsers/parser/PrismParser.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm/models/sparse/Mdp.h"
#include "stochastic-windows/directfixedwindow/DirectFixedWindowObjective.h"

namespace  {


    class ValueIterationSolverInitialization {
    public:
        ValueIterationSolverInitialization() {
            auto const& minMaxSettings = storm::settings::getModule<storm::settings::modules::MinMaxEquationSolverSettings>();
            auto minMaxEquationSolvingTechnique = minMaxSettings.getMinMaxEquationSolvingMethod();
            if (minMaxEquationSolvingTechnique != storm::solver::MinMaxMethod::ValueIteration) {
                storm::settings::mutableManager().setFromString("--minmax:method vi");
            }
        }
    };


    template<typename TestInit>
    class WindowObjectiveTest : public ::testing::Test {
    public:
        const double precision = 1e-6;
        WindowObjectiveTest() {
            TestInit();
            // model path
            std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/window_mp_par.nm";

            storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
            storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
            std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
            mdp = model->as<storm::models::sparse::Mdp<double>>();
        }
        std::shared_ptr<storm::models::sparse::Mdp<double>> const& getMDP() {
            return this->mdp;
        }

    private:
        std::shared_ptr<storm::models::sparse::Mdp<double>> mdp;
    };

    typedef ::testing::Types<
            ValueIterationSolverInitialization
    > TestingTypes;

    TYPED_TEST_CASE(WindowObjectiveTest, TestingTypes);

    TYPED_TEST(WindowObjectiveTest, directFixedWindowTest) {
        // all states
        storm::storage::BitVector phiStates(this->getMDP()->getNumberOfStates(), true);
        // Direct Fixed Mean-payoff
        sw::DirectFixedWindow::DirectFixedWindowMeanPayoffObjective<double> dfwMpObjective(*(this->getMDP()), "weights", 3);
        sw::storage::ValuesAndScheduler<double> result = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwMpObjective, true);
        EXPECT_NEAR(0.7596153846, result.values[0], this->precision);
        EXPECT_NEAR(0, result.values[1], this->precision);
        EXPECT_NEAR(result.values[2], result.values[0], this->precision);
        EXPECT_NEAR(1, result.values[3], this->precision);
        EXPECT_NEAR(1, result.values[4], this->precision);
        EXPECT_NEAR(0.03846153846, result.values[5], this->precision);
        EXPECT_NEAR(result.values[3], result.values[6], this->precision);
        EXPECT_NEAR(result.values[4], result.values[7], this->precision);
        EXPECT_NEAR(0.03846153846, result.values[8], this->precision);
        EXPECT_NEAR(0, result.values[9], this->precision);
        EXPECT_NEAR(0.1538461538, result.values[10], this->precision);
        EXPECT_NEAR(1, result.values[11], this->precision);
        // Direct Fixed Parity
        sw::DirectFixedWindow::DirectFixedWindowParityObjective<double> dfwParObjective(*(this->getMDP()), "priorities", 3);
        result = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwParObjective);
        EXPECT_NEAR(0.78125, result.values[0], this->precision);
        EXPECT_NEAR(0.3333333333, result.values[1], this->precision);
        EXPECT_NEAR(0.78125, result.values[2], this->precision);
        EXPECT_NEAR(1, result.values[3], this->precision);
        EXPECT_NEAR(1, result.values[4], this->precision);
        EXPECT_NEAR(0.125, result.values[5], this->precision);
        EXPECT_NEAR(result.values[3], result.values[6], this->precision);
        EXPECT_NEAR(result.values[4], result.values[7], this->precision);
        EXPECT_NEAR(0.25, result.values[8], this->precision);
        EXPECT_NEAR(0, result.values[9], this->precision);
        EXPECT_NEAR(0.3125, result.values[10], this->precision);
        EXPECT_NEAR(1, result.values[11], this->precision);
    }

    TYPED_TEST(WindowObjectiveTest, mecClassificationTest) {
        // Unfoldings
        // Parity
        sw::storage::MaximalEndComponentDecompositionUnfoldingParity<double> unfoldingPar(*this->getMDP(), "priorities", 3);
        // Mean Payoff
        sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<double> unfoldingMp(*this->getMDP(), "weights", 3);

        // expected sets
        storm::storage::BitVector safeStates(12, false);
        storm::storage::BitVector goodStates(12, false);
        safeStates.set(3); safeStates.set(4); safeStates.set(6); safeStates.set(7); safeStates.set(11);
        goodStates.set(1); goodStates.set(3); goodStates.set(4); goodStates.set(6); goodStates.set(7); goodStates.set(11);

        // Classification by unfolding MECs (Par)
        sw::FixedWindow::MaximalEndComponentClassifier<double> classifierUnfoldingPar(*this->getMDP(), unfoldingPar);
        ASSERT_EQ(safeStates, classifierUnfoldingPar.getSafeStateSpace());
        ASSERT_EQ(goodStates, classifierUnfoldingPar.getGoodStateSpace());
        // Classification by unfolding MECs (MP)
        sw::FixedWindow::MaximalEndComponentClassifier<double> classifierUnfoldingMP(*this->getMDP(), unfoldingMp);
        ASSERT_EQ(safeStates, classifierUnfoldingMP.getSafeStateSpace());
        ASSERT_EQ(goodStates, classifierUnfoldingMP.getGoodStateSpace());
        // Classification by considering MECs as games (MP)
        sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<double> games(*this->getMDP(), "weights", 3);
        sw::FixedWindow::MaximalEndComponentClassifier<double> gameClassifier(*this->getMDP(), games);
        ASSERT_EQ(classifierUnfoldingMP.getSafeStateSpace(), gameClassifier.getSafeStateSpace());
        ASSERT_EQ(classifierUnfoldingMP.getGoodStateSpace(), gameClassifier.getGoodStateSpace());
    }

}
