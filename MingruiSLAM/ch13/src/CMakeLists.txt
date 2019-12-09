add_library(myslam SHARED 
                            camera.cpp
                            config.cpp
                            frame.cpp
                            feature.cpp
                            mappoint.cpp
                            dataset.cpp
                            map.cpp
                            )



target_link_libraries(myslam ${THIRD_PARTY_LIBS})