// bypass_sim.c — simulates what a bypass miner would look like
// calls nvidia ioctl directly without going through libcuda
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main() {
    // open the compute device directly
    int fd = open("/dev/nvidia0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/nvidia0");
        return 1;
    }

    printf("pid=%d hammering nvidia0 ioctl without libcuda...\n", getpid());

    // hammer ioctl in a loop — simulates miner behavior
    // cmd 0 is harmless, driver will reject it but ioctl still fires
    for (int i = 0; i < 10000; i++) {
        ioctl(fd, 0, NULL);
        usleep(1000); // 1ms between calls
    }

    close(fd);
    return 0;
}
