add_executable(treearray_test_np treearray.c treearray_test.c)
if (NOT WIN32)
    target_link_libraries(treearray_test_np m)
endif()
