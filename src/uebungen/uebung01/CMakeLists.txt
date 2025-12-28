set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/u01)

add_executable(rechteck rechteck.c)
install(TARGETS rechteck DESTINATION bin/u01)

add_executable(datum datum.c)
install(TARGETS datum DESTINATION bin/u01)

add_executable(eingabe eingabe.c)
install(TARGETS eingabe DESTINATION bin/u01)
if (NOT WIN32)
    target_link_libraries(eingabe m) # math
endif()

add_executable(algorithmen algorithmen.c)
install(TARGETS algorithmen DESTINATION bin/u01)
if (NOT WIN32)
    target_link_libraries(algorithmen m) # math
endif()


add_executable(01_02 01_02.c)
add_executable(01_03 01_03.c)
add_executable(01_04a 01_04a.c)
target_link_libraries(01_04a m)
add_executable(01_04b 01_04b.c)
target_link_libraries(01_04b m)
add_executable(01_04c 01_04c.c)
add_executable(01_04d 01_04d.c)
add_executable(01_05a 01_05a.c)
add_executable(01_05b 01_05b.c)
add_executable(01_05c 01_05c.c)
target_link_libraries(01_05c m)
add_executable(01_05d 01_05d.c)
target_link_libraries(01_05d m)