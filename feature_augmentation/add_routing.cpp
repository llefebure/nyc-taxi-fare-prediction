#include "osrm/match_parameters.hpp"
#include "osrm/nearest_parameters.hpp"
#include "osrm/route_parameters.hpp"
#include "osrm/table_parameters.hpp"
#include "osrm/trip_parameters.hpp"

#include "osrm/coordinate.hpp"
#include "osrm/engine_config.hpp"
#include "osrm/json_container.hpp"

#include "osrm/osrm.hpp"
#include "osrm/status.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <boost/algorithm/string.hpp>

#include <cstdlib>

int main(int argc, const char *argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " data.osrm\n" << " coord_pairs.csv";
        return EXIT_FAILURE;
    }
    
    // Read in coordinates
    std::vector<std::vector<std::string>> routePairs;
    std::ifstream file(argv[2]);
    std::string value;
    while (getline(file, value)) {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, value, boost::is_any_of(",;"));
        routePairs.push_back(vec);
    }

    using namespace osrm;
    
    // Configure based on a .osrm base path, and no datasets in shared mem from osrm-datastore
    EngineConfig config;

    config.storage_config = {argv[1]};
    config.use_shared_memory = false;

    // We support two routing speed up techniques:
    // - Contraction Hierarchies (CH): requires extract+contract pre-processing
    // - Multi-Level Dijkstra (MLD): requires extract+partition+customize pre-processing
    //
    config.algorithm = EngineConfig::Algorithm::CH;
    // config.algorithm = EngineConfig::Algorithm::MLD;

    // Routing machine with several services (such as Route, Table, Nearest, Trip, Match)
    const OSRM osrm{config};
 
    std::ofstream route_file;
    route_file.open("route_info.txt");
    for (int i = 0; i < routePairs.size(); i++) {
        // The following shows how to use the Route service; configure this service
        RouteParameters params;

        // Route in monaco
        params.coordinates.push_back({util::FloatLongitude{std::stod(routePairs[i][0])}, util::FloatLatitude{std::stod(routePairs[i][1])}});
            params.coordinates.push_back({util::FloatLongitude{std::stod(routePairs[i][2])}, util::FloatLatitude{std::stod(routePairs[i][3])}});
        params.steps = true;

        // Response is in JSON format
        json::Object result;

        // Execute routing request, this does the heavy lifting
        const auto status = osrm.Route(params, result);
        if (status == Status::Ok)
        {
            auto &routes = result.values["routes"].get<json::Array>();
            // Let's just use the first route
            auto &route = routes.values.at(0).get<json::Object>();
            auto leg = route.values["legs"].get<json::Array>().values.at(0).get<json::Object>();
            
            const auto distance = route.values["distance"].get<json::Number>().value;
            const auto duration = route.values["duration"].get<json::Number>().value;
            
            // Warn users if extract does not contain the default coordinates from above
            if (distance == 0 || duration == 0)
            {
                std::cout << "Note: distance or duration is zero. ";
                std::cout << "You are probably doing a query outside of the OSM extract.\n\n";
            }
            
            route_file <<  distance << ",";
            route_file << duration << ",";
            route_file << "\"" << leg.values["summary"].get<json::String>().value << "\"" << std::endl;
        }
        else if (status == Status::Error) {
            const auto code = result.values["code"].get<json::String>().value;
            const auto message = result.values["message"].get<json::String>().value;

            std::cout << "Code: " << code << "\n";
            std::cout << "Message: " << code << "\n";
        }
    }
    route_file.close();
    return EXIT_SUCCESS;
}
