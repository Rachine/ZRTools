#include "cosine.h"
#include <tgmath.h>
#include <stdio.h>


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
