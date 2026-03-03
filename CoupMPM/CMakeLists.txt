# CMakeLists.txt for the COUPMPM package
# Drop this file alongside the sources in lammps/src/COUPMPM/

target_sources(lammps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/atom_vec_mpm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fix_coupmpm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fix_coupmpm_contact.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fix_coupmpm_cohesive.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fix_coupmpm_adaptivity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fix_coupmpm_output.cpp
)

target_include_directories(lammps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
