//
// Created by florent on 23/09/19.
//

#include <storm/models/sparse/Mdp.h>
#include "gtest/gtest.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm-parsers/parser/NondeterministicModelParser.h"
#include "stochastic-windows/util/Graphviz.h"

TEST(SchedulerTest, meanPayoffWindowGameTest) {
    std::string tra_file = STORM_TEST_RESOURCES_DIR "/tra/dfwMPGame.tra";
    std::string rew_file = STORM_TEST_RESOURCES_DIR "/rew/dfwMPGame.trans.rew";
    std::string lab_file = STORM_TEST_RESOURCES_DIR "/lab/dfwMPGame.lab";
    std::string choicelab_file = STORM_TEST_RESOURCES_DIR "/lab/dfwMPGame.choicelab";
    storm::models::sparse::Mdp<double> mdp(storm::parser::NondeterministicModelParser<>::parseMdp(tra_file, lab_file, "", rew_file, choicelab_file));
    storm::models::sparse::StandardRewardModel<double> rewardModel = mdp.getUniqueRewardModel();
    rewardModel.reduceToStateBasedRewards(mdp.getTransitionMatrix());
    mdp.addRewardModel("weights", storm::models::sparse::StandardRewardModel<double>(boost::none, rewardModel.getStateActionRewardVector()));
    // mdp.restrictRewardModels({"weights"});
    sw::util::graphviz::GraphVizBuilder::mdpGraphExport(mdp, "", "weights");

}
