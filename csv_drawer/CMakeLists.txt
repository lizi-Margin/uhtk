message("Enter subdirectory csv_drawer")
add_executable(test_csv_drawer 
    test_csv_drawer.cpp
    csv_drawer.h
    ../rubbish_can.h
)

# set_target_properties(test_csv_drawer PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )
