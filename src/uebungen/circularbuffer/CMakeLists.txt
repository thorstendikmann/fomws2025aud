#add_executable(circularbuffer_np circularbuffer.c)
set(SOURCES circularbuffer.c circularbuffer.h)
add_executable(circularbuffer_test_np circularbuffer_test.c circularbuffer.c)
#target_link_libraries(circularbuffer_test_np circularbuffer.h)