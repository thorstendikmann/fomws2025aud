add_executable(treeptr_test_np treeptr.c treeptr_test.c)
if (NOT WIN32)
    target_link_libraries(treeptr_test_np m)
endif()
