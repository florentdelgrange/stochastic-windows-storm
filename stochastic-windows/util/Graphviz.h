//
// Created by Florent Delgrange on 2018-11-20.
//

#include "storm/models/sparse/Mdp.h"
#include "storm/storage/SparseMatrix.h"
#include <boost/graph/adj_list_serialize.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <boost/property_map/transform_value_property_map.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/variant.hpp>
#include <fstream>
#include <stochastic-windows/ECsUnfolding.h>

#ifndef STORM_GRAPHVIZ_H
#define STORM_GRAPHVIZ_H

namespace sw {
    namespace util {
        namespace graphviz {

            using namespace boost;
            namespace Nodes {
                struct State { std::string id; };
                struct Action { std::string id; double weight = 0; };

                static inline std::ostream& operator<<(std::ostream& os, State const& s) { return os << s.id; }
                static inline std::ostream& operator<<(std::ostream& os, Action const& a) { return os << "a" << a.id; }

                std::string id_of(State const& s) { return "state" + s.id; }
                std::string id_of(Action const& a) { return "action" + a.id; }
                std::string label_of(State const& s) { return ""; }
                std::string label_of(Action const &a) {
                    if (a.weight != 0.)
                        return boost::lexical_cast<std::string>(a.weight) + "  ";
                    else
                        return "";
                }
                std::string shape_of(State const& s) { return "circle"; }
                std::string shape_of(Action const& a) { return "point"; }
            }
            namespace Transitions {
                struct Choice { double weight = 0; };
                struct ProbabilityTransition { double probability; };

                static inline std::ostream& operator<<(std::ostream& os, Choice const& c) {
                    if (c.weight != 0.) return os << c.weight; else return os << "";
                }
                static inline std::ostream& operator<<(std::ostream& os, ProbabilityTransition const& p) {
                    if (p.probability < 1)
                        return os << p.probability;
                    else
                        return os << "";
                }
            }

            using Vertex = boost::variant<Nodes::State, Nodes::Action>;
            using Edge = boost::variant<Transitions::Choice, Transitions::ProbabilityTransition>;

            std::string id_of(Vertex const &v) {
                return boost::apply_visitor([](auto const &node) { return Nodes::id_of(node); }, v);
            }

            std::string shape_of(Vertex const &v) {
                return boost::apply_visitor([](auto const &node) { return Nodes::shape_of(node); }, v);
            }

            std::string label_of(Vertex const &v) {
                return boost::apply_visitor([](auto const &node) { return Nodes::label_of(node); }, v);
            }

            class GraphVizBuilder {
            public:
                /**
                 * Export the MDP (encoded as a sparse matrix) to a dot file depicting the MDP
                 * @param matrix
                 * @param rewardVector
                 * @param graphName
                 * @param stateNames
                 */
                static void mdpGraphExport(storm::storage::SparseMatrix<double> const &matrix,
                                           std::vector<double> rewardVector = std::vector<double>(),
                                           std::string graphName = "mdp",
                                           std::string outputDir = "/tmp",
                                           std::vector<std::string> stateNames = std::vector<std::string>()) {

                    if (rewardVector.empty()) {
                        std::vector<double> zeroRewards(matrix.getRowCount(), 0.);
                        rewardVector.reserve(zeroRewards.size());
                        rewardVector.insert(rewardVector.end(), zeroRewards.begin(), zeroRewards.end());
                    } else
                        assert(rewardVector.size() == matrix.getRowCount());

                    if (stateNames.empty()) {
                        std::vector<std::string> noName(matrix.getRowGroupCount(), "");
                        stateNames.reserve(noName.size());
                        stateNames.insert(stateNames.end(), noName.begin(), noName.end());
                    } else
                        assert(stateNames.size() == matrix.getRowGroupCount());

                    typedef adjacency_list<vecS, vecS, directedS, Vertex, Edge> Graph;

                    Graph g;
                    uint_fast64_t nextState;
                    double p;
                    std::vector<uint_fast64_t> groups = matrix.getRowGroupIndices();
                    std::vector<Graph::vertex_descriptor> stateVertices(matrix.getRowGroupCount());
                    std::vector<Graph::vertex_descriptor> actionVertices(matrix.getRowCount());

                    for (uint_fast64_t state = 0; state < matrix.getRowGroupCount(); ++state) {
                        Graph::vertex_descriptor s;
                        if (stateNames[state] == "")
                            s = add_vertex(Nodes::State{'s' + std::to_string(state)}, g);
                        else
                            s = add_vertex(Nodes::State{stateNames[state]}, g);
                        stateVertices[state] = s;
                        for (uint_fast64_t row = groups[state]; row < groups[state + 1]; ++row) {
                            Graph::vertex_descriptor a = add_vertex(
                                    Nodes::Action{std::to_string(row), rewardVector[row]},
                                    g);
                            add_edge(s, a, Transitions::Choice{}, g);
                            actionVertices[row] = a;
                        }
                    }

                    for (uint_fast64_t state = 0; state < matrix.getRowGroupCount(); ++state) {
                        for (uint_fast64_t row = groups[state]; row < groups[state + 1]; ++row) {
                            for (auto entry : matrix.getRow(row)) {
                                nextState = entry.getColumn();
                                p = entry.getValue();
                                add_edge(actionVertices[row], stateVertices[nextState],
                                         Transitions::ProbabilityTransition{p}, g);
                            }
                        }
                    }

                    {
                        std::ofstream dot_file(outputDir + "/" + graphName + ".dot");
                        boost::dynamic_properties dp;

                        dp.property("node_id", boost::make_transform_value_property_map(&sw::util::graphviz::id_of,
                                                                                        boost::get(boost::vertex_bundle,
                                                                                                   g)));
                        dp.property("shape", boost::make_transform_value_property_map(&sw::util::graphviz::shape_of,
                                                                                      boost::get(boost::vertex_bundle,
                                                                                                 g)));
                        dp.property("label", boost::make_transform_value_property_map(
                                [](Vertex const &v) { return boost::lexical_cast<std::string>(v); },
                                boost::get(boost::vertex_bundle, g)));
                        dp.property("xlabel", boost::make_transform_value_property_map(&sw::util::graphviz::label_of,
                                                                                       boost::get(boost::vertex_bundle,
                                                                                                  g)));
                        dp.property("label", boost::make_transform_value_property_map(
                                [](Edge const &e) { return boost::lexical_cast<std::string>(e); },
                                boost::get(boost::edge_bundle, g)));

                        write_graphviz_dp(dot_file, g, dp);
                    }
                }

                static void bndGWMPUnfoldedECsExport(sw::BndGoodWindowMP::ECsUnfolding<double> &unfoldedECs,
                                                     std::string graphName = "mdp",
                                                     std::string outputDir = "/tmp") {

                    for (uint_fast64_t k = 1; k <= unfoldedECs.getNumberOfUnfoldedECs(); ++ k) {
                        std::vector<sw::BndGoodWindowMP::StateWeightWindowLength<double>>
                            newStatesMeaning = unfoldedECs.getNewStatesMeaning(k);
                        std::vector<std::string> stateNames = std::vector<std::string>(newStatesMeaning.size());
                        // the state with index 0 in the unfolding is the sink state
                        stateNames[0] = "⊥";
                        for (uint_fast64_t i = 1; i < newStatesMeaning.size(); ++ i) {
                            std::ostringstream stream;
                            stream << "(s" << newStatesMeaning[i].state << ", " <<
                                newStatesMeaning[i].currentSumOfWeights << ", " <<
                                newStatesMeaning[i].currentWindowLength << ")";
                            stateNames[i] = stream.str();
                        }
                        std::ostringstream stream;
                        stream << graphName << "_unfoldedEC_" << k;
                        mdpGraphExport(unfoldedECs.getUnfoldedMatrix(k), std::vector<double>(), stream.str(), outputDir,
                                       stateNames);
                    }
                }
            };
        }
    }
}
#endif //STORM_GRAPHVIZ_H
