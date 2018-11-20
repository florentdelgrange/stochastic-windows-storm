//
// Created by Florent Delgrange on 2018-11-20.
//

#include "storm/models/sparse/Mdp.h"
#include "storm/storage/SparseMatrix.h"
#include "fstream"
#include "boost/graph/adjacency_list.hpp"
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
    struct Action { std::string id; };

    static inline std::ostream& operator<<(std::ostream& os, State const& b) { return os << "State " << b.id; }
    static inline std::ostream& operator<<(std::ostream& os, Action const& b) { return os << "Action " << b.id; }

    std::string id_of(State const& b) { return b.id; }
    std::string id_of(Action const& c) { return c.id; }
    std::string shape_of(State const& b) { return "circle"; }
    std::string shape_of(Action const& c) { return "dot"; }
}

using Nodes::State;
using Nodes::Action;
using Vertex = boost::variant<State, Action>;

namespace sw {
    namespace util {
        class GraphViz {
        public:

            std::string id_of(Vertex const& v) {
                return boost::apply_visitor([](auto const& node) { return Nodes::id_of(node); }, v);
            }
            std::string shape_of(Vertex const& v) {
                return boost::apply_visitor([](auto const& node) { return Nodes::shape_of(node); }, v);
            }

            void mdpGraphExport(std::shared_ptr<storm::models::sparse::Mdp<double>> mdp,
                    std::string const& rewardModelName,
                    std::string graphName=""){

                mdp->printModelInformationToStream(std::cout);
                storm::storage::SparseMatrix<double> matrix = mdp->getTransitionMatrix();
                struct Edge { double proba; };
                typedef adjacency_list<vecS, vecS, directedS, Vertex, Edge> Graph;

                Graph g;

                uint_fast64_t nextState;
                double p;
                std::vector<uint_fast64_t> groups = matrix.getRowGroupIndices();
                std::vector<Graph::vertex_descriptor> stateVertices(matrix.getRowGroupCount());
                std::vector<Graph::vertex_descriptor> actionVertices(matrix.getRowCount());

                for (uint_fast64_t state = 0; state < matrix.getRowGroupCount(); ++ state) {
                    Graph::vertex_descriptor s = add_vertex(Nodes::State{ std::to_string(state) }, g);
                    stateVertices[state] = s;
                    for (uint_fast64_t row = groups[state]; row < groups[state + 1]; ++ row) {
                        Graph::vertex_descriptor a = add_vertex(Nodes::Action{ std::to_string(row) }, g);
                        add_edge(s, a, g);
                        actionVertices[row] = a;
                    }
                }

                for (uint_fast64_t state = 0; state < matrix.getRowGroupCount(); ++ state) {
                    for (uint_fast64_t row = groups[state]; row < groups[state + 1]; ++ row) {
                        for (auto entry : matrix.getRow(row)) {
                            nextState = entry.getColumn();
                            p = entry.getValue();
                            add_edge(stateVertices[state], actionVertices[row], {p}, g);
                        }
                    }
                }

                std::ofstream outf("/tmp/min.gv");
                boost::dynamic_properties dp;

                dp.property("node_id", boost::make_transform_value_property_map(Nodes::id_of, boost::get(boost::vertex_bundle, g)));
                dp.property("shape", boost::make_transform_value_property_map(Nodes::shape_of, boost::get(boost::vertex_bundle, g)));
                dp.property("label", boost::make_transform_value_property_map(
                        [](Vertex const& v) { return boost::lexical_cast<std::string>(v); },
                        boost::get(boost::vertex_bundle, g)));
                dp.property("label", get(&Edge::proba, g));
            }
        };

    }
}