#include <tgmath.h>
#include <stdio.h>
#include <stdbool.h>

typedef unsigned char byte;

const static int BITS_IN_[256] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

float approximate_cosine (byte* x, byte* y, int SIG_NUM_BYTES);

bool signature_is_zeroed (byte* x, int SIG_NUM_BYTES);

int hamming (byte* x, byte* y, int SIG_NUM_BYTES) {
  int diff = 0;
  for (int i = 0; i < SIG_NUM_BYTES; i++)
    diff += BITS_IN_[x[i] ^ y[i]];
  return diff;
}

int numComparisons = 0;

float approximate_cosine (byte* x, byte* y, int SIG_NUM_BYTES) {
  numComparisons++;
  return cos(hamming(x,y, SIG_NUM_BYTES) * 3.1415926535897932384626433832795029/ (SIG_NUM_BYTES*8));
}

bool signature_is_zeroed (byte* x, int SIG_NUM_BYTES) {
  for (int i = 0; i < SIG_NUM_BYTES; i++)
    if (x[i] != 0)
      return false;
  return true;
}
