#include <stdio.h>
#include <stdlib.h>

#include "circularbuffer.h"

int main(void)
{
    printf("- %s, %d\n", __func__, __LINE__);

    CircularBuffer c;
    int MAX_SIZE = 10;
    circular_buffer_init(&c, MAX_SIZE);

    circular_buffer_push_back(&c, 'a');
    circular_buffer_print(&c);

    circular_buffer_push_back(&c, 'b');
    circular_buffer_print(&c);

    char pop = circular_buffer_pop_back(&c);
    circular_buffer_print(&c);
    printf("popped back %c\n", pop);

    circular_buffer_push_back(&c, 'c');
    circular_buffer_print(&c);

    pop = circular_buffer_pop_front(&c);
    circular_buffer_print(&c);
    printf("popped front %c\n", pop);

    circular_buffer_push_back(&c, 'd');
    circular_buffer_print(&c);

    circular_buffer_destroy(&c);

    return 0;
}
