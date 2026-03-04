# ==============================================================================
# CMakeLists.txt for the COUPMPM package
# ==============================================================================

# Extract the package name from the directory name (COUPMPM)
get_filename_component(PKG ${CMAKE_CURRENT_LIST_DIR} NAME)

# Only compile these files if the user enables the package at configure time
if(PKG_${PKG})
  
  # Automatically discover all C++ source and header files in this directory
  file(GLOB ${PKG}_SOURCES *.cpp)
  file(GLOB ${PKG}_HEADERS *.h)
  
  # Append them to the master LAMMPS build lists
  list(APPEND LAMMPS_SOURCES ${${PKG}_SOURCES})
  list(APPEND LAMMPS_HEADERS ${${PKG}_HEADERS})
  
  # (Optional) If CoupMPM ever requires external libraries like Eigen or VTK 
  # in Phase 2 or 3, you would link them right here:
  # target_link_libraries(lammps PRIVATE some_external_lib)

endif()
