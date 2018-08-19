# Feature Augmentation

This details how to setup the [osrm project](https://github.com/Project-OSRM/osrm-backend) to do offline routing. This is used to generate additional route level features.

### Building `osrm-backend`

The first step is to build `osrm-backend` from source. See [here](https://github.com/Project-OSRM/osrm-backend#building-from-source) for details. I did this on a Google Cloud VM instance.

### Downloading the OSM Extract

You'll need to download an appropriate OSM extract from a site such as [Geofabrik](http://download.geofabrik.de/). I downloaded the US Northeast region for this task.

### Preprocess the OSM Extract

Next, you'll need to preprocess the OSM extract for routing using `osrm-extract`, `osrm-partition`, and `osrm-customize`. See [here](https://github.com/Project-OSRM/osrm-backend/wiki/Running-OSRM#quickstart).

### Run

Finally, we can compile `add_routing.cpp` and run.

```
mkdir build
cd build
cmake ..
make
./osrm-routing-from-file [osrm extract] [train/test.csv]
```

The last command will generate an `out.txt``file with additional features for each key.
