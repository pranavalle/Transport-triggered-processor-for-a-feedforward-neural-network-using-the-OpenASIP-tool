
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define numInputs 3
#define numTrainingSets 8

void generateRandomInputs(const char *fileName) {
    srand(time(NULL));
    FILE *file = fopen(fileName, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < numTrainingSets; i++) {
        double area = (double)(rand() % 3000 + 1000); // Generating random area between 1000 and 4000 sq.ft
        double bedrooms = (double)(rand() % 5);       // Generating random number of bedrooms between 0 and 4
        double parking = (double)(rand() % 4);        // Generating random parking spots between 0 and 3

        fprintf(file, "%g %g %g\n", area, bedrooms, parking);
    }

    fclose(file);
}

int main(void) {
    generateRandomInputs("tta_stream_v1.in");
    return 0;
}

