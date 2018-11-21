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

#ifndef STORM_GRAPHVIZ_H
#define STORM_GRAPHVIZ_H

#endif //STORM_GRAPHVIZ_H

using namespace boost;
namespace Nodes {
    struct State { std::string id; };
    struct Action { std::string id; double weight = 0; };

    static inline std::ostream& operator<<(std::ostream& os, State const& s) { return os << "s" << s.id; }
    static inline std::ostream& operator<<(std::ostream& os, Action const& a) { return os << "a" << a.id << " | " << a.weight; }

    std::string id_of(State const& s) { return "state" + s.id; }
    std::string id_of(Action const& a) { return "action" + a.id; }
    std::string label_of(State const& s) { return ""; }
    std::string label_of(Action const &a) {
        if (a.weight != 0.)
            return boost::lexical_cast<std::string>(a);
        else
            return "a" + a.id;
    }
    std::string shape_of(State const& s) { return "circle"; }
    std::string shape_of(Action const& a) { return "point"; }
}
namespace Transitions {
    struct Choice {};
    struct ProbabilityTransition { double probability; };

    static inline std::ostream& operator<<(std::ostream& os, Choice const& t) { return os << ""; }
    static inline std::ostream& operator<<(std::ostream& os, ProbabilityTransition const& p) {
        if (p.probability < 1)
            return os << p.probability;
        else
            return os << "";
    }
}

using Vertex = boost::variant<Nodes::State, Nodes::Action>;
using Edge = boost::variant<Transitions::Choice, Transitions::ProbabilityTransition>;

namespace sw {
    namespace util {

        std::string id_of(Vertex const &v) {
            return boost::apply_visitor([](auto const &node) { return Nodes::id_of(node); }, v);
        }

        std::string shape_of(Vertex const &v) {
            return boost::apply_visitor([](auto const &node) { return Nodes::shape_of(node); }, v);
        }

        std::string label_of(Vertex const &v) {
            return boost::apply_visitor([](auto const &node) { return Nodes::label_of(node); }, v);
        }

        class GraphViz {
        public:

            static void mdpGraphExport(storm::storage::SparseMatrix<double> const &matrix,
                                       std::vector<double> rewardVector = std::vector<double>(),
                                       std::string graphName,
                                       std::vector<std::string> stateNames = std::vector<std::string>()) {

                assert(rewardVector.size() == matrix.getRowCount());
                if (rewardVector.empty()) {
                    std::vector<double> zeroRewards(matrix.getRowCount(), 0.);
                    rewardVector.reserve(zeroRewards.size());
                    rewardVector.insert(rewardVector.end(), zeroRewards.begin(), zeroRewards.end());
                }

                assert(stateNames.size() == matrix.getRowGroupCount());
                if (stateNames.empty()) {
                    std::vector<std::string> noName(matrix.getRowGroupCount(), "");
                    stateNames.reserve(noName.size());
                    stateNames.insert(stateNames.end(), noName.begin(), noName.end());
                }

                typedef adjacency_list<vecS, vecS, directedS, Vertex, Edge> Graph;

                Graph g;
                uint_fast64_t nextState;
                double p;
                std::vector<uint_fast64_t> groups = matrix.getRowGroupIndices();
                std::vector<Graph::vertex_descriptor> stateVertices(matrix.getRowGroupCount());
                std::vector<Graph::vertex_descriptor> actionVertices(matrix.getRowCount());

                for (uint_fast64_t state = 0; state < matrix.getRowGroupCount(); ++state) {
                    Graph::vertex_descriptor s = add_vertex(Nodes::State{std::to_string(state)}, g);
                    stateVertices[state] = s;
                    for (uint_fast64_t row = groups[state]; row < groups[state + 1]; ++row) {
                        Graph::vertex_descriptor a = add_vertex(Nodes::Action{std::to_string(row), rewardVector[row]},
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
                    std::ofstream dot_file("/tmp/" + graphName + ".dot");
                    boost::dynamic_properties dp;

                    dp.property("node_id", boost::make_transform_value_property_map(&sw::util::id_of,
                                                                                    boost::get(boost::vertex_bundle,
                                                                                               g)));
                    dp.property("shape", boost::make_transform_value_property_map(&sw::util::shape_of,
                                                                                  boost::get(boost::vertex_bundle, g)));
                    dp.property("label", boost::make_transform_value_property_map(
                            [](Vertex const &v) { return boost::lexical_cast<std::string>(v); },
                            boost::get(boost::vertex_bundle, g)));
                    dp.property("xlabel", boost::make_transform_value_property_map(&sw::util::label_of,
                                                                                   boost::get(boost::vertex_bundle,
                                                                                              g)));
                    dp.property("label", boost::make_transform_value_property_map(
                            [](Edge const &e) { return boost::lexical_cast<std::string>(e); },
                            boost::get(boost::edge_bundle, g)));

                    write_graphviz_dp(dot_file, g, dp);
                }

            }
        };
    }
}