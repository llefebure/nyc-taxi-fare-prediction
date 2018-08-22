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
#include <boost/algorithm/string/join.hpp>

#include <cstdlib>

int main(int argc, const char *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " data.osrm coord_pairs.csv outfile_name.csv";
        return EXIT_FAILURE;
    }
    
    using namespace osrm;

    // Configure based on a .osrm base path, and no datasets in shared mem from osrm-datastore
    EngineConfig config;

    config.storage_config = {argv[1]};
    config.use_shared_memory = false;

    // We support two routing speed up techniques:
    // - Contraction Hierarchies (CH): requires extract+contract pre-processing
    // - Multi-Level Dijkstra (MLD): requires extract+partition+customize pre-processing
    config.algorithm = EngineConfig::Algorithm::MLD;

    // Routing machine with several services (such as Route, Table, Nearest, Trip, Match)
    const OSRM osrm{config};

    // Open file, and parse header
    std::ifstream file(argv[2]);
    std::vector<std::string> header_items;
    std::string header;
    getline(file, header);
    if (header[header.size() - 1] == '\r') header.pop_back(); // Pop off carriage return
    boost::algorithm::split(header_items, header, boost::is_any_of(","));
    int key, lat1, lon1, lat2, lon2;
    for (int i = 0; i < header_items.size(); i++){
        if (header_items[i] == "key") key = i;
        if (header_items[i] == "pickup_latitude") lat1 = i;
        if (header_items[i] == "pickup_longitude") lon1 = i;
        if (header_items[i] == "dropoff_latitude") lat2 = i;
        if (header_items[i] == "dropoff_longitude") lon2 = i;
    }

    // Open output file and write header line
    std::ofstream route_file;
    route_file.open(argv[3]);
    for (int i = 0; i < header_items.size(); i++) {
        route_file << header_items[i] << ",";
    }
    route_file << "distance,duration,summary" << std::endl;
    
    // Main loop
    int i = 0;
    std::string value;
    while (getline(file, value)) {
        i += 1;
        if (i % 100000 == 0) std::cout << "Processed " << i << " rows" << std::endl;

        // Parse row
        std::vector<std::string> row;
        if (value[value.size() - 1] == '\r') value.pop_back(); // Pop off carriage return
        boost::algorithm::split(row, value, boost::is_any_of(","));

        if (row[lon1] == "" || row[lat1] == "" || row[lon2] == "" || row[lat2] == "") {
            // lat,lon is malformed, so don't attempt routing
            for (int j = 0; j < row.size(); j++) { // Write full row
                route_file << row[j] << ",";
            }
            route_file << ",,\"\"" << std::endl;
        } else {
            RouteParameters params;
            params.coordinates.push_back({util::FloatLongitude{std::stod(row[lon1])}, util::FloatLatitude{std::stod(row[lat1])}});
            params.coordinates.push_back({util::FloatLongitude{std::stod(row[lon2])}, util::FloatLatitude{std::stod(row[lat2])}});
            params.steps = true;
                
            // Response is in JSON format
            json::Object result;

            // Execute routing request, this does the heavy lifting
            const auto status = osrm.Route(params, result);
            if (status == Status::Ok) {
                auto &routes = result.values["routes"].get<json::Array>();
                // Let's just use the first route
                auto &route = routes.values.at(0).get<json::Object>();
                auto leg = route.values["legs"].get<json::Array>().values.at(0).get<json::Object>();
                
                const auto distance = route.values["distance"].get<json::Number>().value;
                const auto duration = route.values["duration"].get<json::Number>().value;
                const auto summary = leg.values["summary"].get<json::String>().value;
                for (int j = 0; j < row.size(); j++) { // Write full row
                    route_file << row[j] << ",";
                }
                route_file << distance << "," << duration << "," << "\"" << summary << "\"" << std::endl;
            }
            else if (status == Status::Error) {
                // Some coordinates may be malformed in the input file
                for (int j = 0; j < row.size(); j++) { // Write full row
                    route_file << row[j] << ",";
                }
                route_file << ",,\"\"" << std::endl;
            }
        }
        if (i % 100000 == 0) route_file.flush();
    }
    route_file.close();
    return EXIT_SUCCESS;
}
