#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../ZRTools/plebdisc/dot.h"
#include "../ZRTools/plebdisc/feat.h"
#include "../ZRTools/plebdisc/signature.h"
#include "../ZRTools/plebdisc/util.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include "second_pass.h"
#define MAXMATCHES 100000
#define MAXDOTS_MF 50000000
/* #include "second_pass" */


struct signature * signature_from_feats_array(PyArrayObject* feats) {
  unsigned char * arr = (unsigned char *) PyArray_DATA(feats);
  int N = PyArray_DIM(feats, 0);
  struct signature * res;
  res = (struct signature *) MALLOC(N*sizeof(struct signature));
  for (int i=0; i<N; i++) {
    res[i].byte_ = arr + (i * feats->dimensions[1]);
    res[i].id = i;
    res[i].query = 0;
  }
  return res;
} 


PyArrayObject* compute_similarity_matrix(PyArrayObject* py_cumsum, PyArrayObject* py_feats1, PyArrayObject* py_feats2, int N1, int N2, int diffspeech, int P, int B, int D, float T) {
/* void compute_similarity_matrix(PyArrayObject* py_radixdots, PyArrayObject* py_cumsum, PyArrayObject* py_feats1, PyArrayObject* py_feats2, int N1, int N2, int diffspeech, int P, int B, int D, float T) { */

  SIG_NUM_BYTES = 8;
  struct signature * feats1 = signature_from_feats_array(py_feats1);
  struct signature * feats2 = signature_from_feats_array(py_feats2);
  
   int Nmax = max(N1,N2);
   int compfact = diffspeech + 1;

   // Initialize the pleb permutations
   initialize_permute();

   // Compute the rotated dot plot
   long maxdots;
   maxdots = P*B*(Nmax)*(2*D+1);

   fprintf(stderr,"Computing sparse dot plot (Max Dots = %ld) ...\n", maxdots); tic();

   Dot *dotlist = (Dot *) MALLOC(maxdots*sizeof(Dot));
   memset(dotlist, 0, maxdots*sizeof(Dot));
   
   int dotcnt;
   dotcnt = pleb( feats1, N1, feats2, N2, diffspeech,
		  P, B, T, D, dotlist );

   fprintf(stderr, "    Total elements in thresholded sparse: %d\n", dotcnt);
   fprintf(stderr, "Finished: %f sec.\n",toc());
   
   // Sort dots by row
   fprintf(stderr, "Applying radix sort of dotlist: "); tic();


   // Creating structured array:
   // http://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c
   /* Py_Initialize(); */
   /* import_array(); */
   PyObject* op = Py_BuildValue("[(s, s), (s, s)]", "xp", "i4", "val", "f4");
   PyArray_Descr* descr;
   PyArray_DescrConverter(op, &descr);
   Py_DECREF(op);
   npy_intp dims[] = {dotcnt};
   PyArrayObject * py_radixdots = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, dims, descr);
   DotXV *radixdots = (DotXV *)PyArray_DATA(py_radixdots);

   /* DotXV *radixdots = (DotXV *)MALLOC( dotcnt*sizeof(DotXV)); */
   
   /* int *cumsum = (int*)MALLOC((compfact*Nmax+1)*sizeof(int)); */
   int * cumsum = (int *) PyArray_DATA(py_cumsum);
   radix_sorty(compfact*Nmax, dotlist, dotcnt, radixdots, cumsum);
   fprintf(stderr, "%f s\n",toc());

   // Sort rows by column
   fprintf(stderr, "Applying qsort of radix bins: "); tic();
   quick_sortx(compfact*Nmax, radixdots, dotcnt, cumsum);
   fprintf(stderr, "%f s\n",toc());

   // Remove duplicate dots in each row
   fprintf(stderr, "Removing duplicate dots from radix bins: "); tic();
   dotcnt = dot_dedup(radixdots, cumsum, compfact*Nmax, T);
   fprintf(stderr, "%f s\n",toc());
   fprintf(stderr, "    Total elements after dedup: %d\n", dotcnt);

   /* // Numpy C-API is not understandable, so we're modifying the */
   /* // attributes directly: */
   /* py_radixdots->data = radixdots;  // data pointer */
   /* py_radixdots->dimensions[0] = dotcnt;  // shape[0] */
   /* PyArray_ENABLEFLAGS(py_radixdots, NPY_ARRAY_OWNDATA);  // change data ownership */

   FREE(feats1);
   FREE(feats2);
   FREE(dotlist);

   return py_radixdots;
}


PyArrayObject* filter_matrix(
      PyArrayObject * Py_cumsum_mf, PyArrayObject * Py_hough,
      PyArrayObject * Py_radixdots, PyArrayObject * Py_cumsum,
      int diffspeech, int Nmax, int dx, int dy, float medthr) {

  DotXV * radixdots = (DotXV *) PyArray_DATA(Py_radixdots);
  int * cumsum = (int *) PyArray_DATA(Py_cumsum);
  int compfact = diffspeech + 1;
  int dotcnt = PyArray_DIM(Py_radixdots, 0);
  
   // Apply the median filter in the X direction
   fprintf(stderr, "Applying median filter to sparse matrix: "); tic();
   Dot *dotlist_mf = (Dot *) MALLOC(MAXDOTS_MF*sizeof(DotXV));
   dotcnt = median_filtx(compfact*Nmax, radixdots, dotcnt, cumsum, dx, medthr, dotlist_mf);
   fprintf(stderr, "%f s\n",toc());
   fprintf(stderr, "    Total elements in filtered sparse: %d\n", dotcnt);

   // Sort mf dots by row
   fprintf(stderr,"Applying radix sort of dotlist_mf: "); tic();

   PyObject* op = Py_BuildValue("[(s, s), (s, s)]", "xp", "i4", "val", "f4");
   PyArray_Descr* descr;
   PyArray_DescrConverter(op, &descr);
   Py_DECREF(op);
   npy_intp dims[] = {dotcnt};
   PyArrayObject* py_radixdots_mf = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, dims, descr);
   DotXV *radixdots_mf = (DotXV *)PyArray_DATA(py_radixdots_mf);

   /* DotXV *radixdots_mf = (DotXV *)MALLOC(dotcnt*sizeof(DotXV)); */
   /* int *cumsum_mf = (int *)MALLOC((compfact*Nmax+1)*sizeof(int)); */
   int * cumsum_mf = (int *) PyArray_DATA(Py_cumsum_mf);
   radix_sorty(compfact*Nmax, dotlist_mf, dotcnt, radixdots_mf, cumsum_mf);
   fprintf(stderr, "%f s\n",toc());

   // Sort mf rows by column
   fprintf(stderr,"Applying qsort of radix_mf bins: "); tic();
   quick_sortx(compfact*Nmax, radixdots_mf, dotcnt, cumsum_mf);
   fprintf(stderr, "%f s\n",toc());

   // Compute the Hough transform
   fprintf(stderr,"Computing hough transform: "); tic();
   /* float *hough = (float*)MALLOC(compfact*Nmax*sizeof(float)); */
   float * hough = (float *) PyArray_DATA(Py_hough);
   hough_gaussy(compfact*Nmax, dotcnt, cumsum_mf, dy, diffspeech, hough);
   fprintf(stderr, "%f s\n",toc());

   /* Py_radixdots_mf->data = radixdots_mf; */
   /* Py_radixdots_mf->dimensions[0] = dotcnt; */
   /* PyArray_ENABLEFLAGS(Py_radixdots_mf, NPY_ARRAY_OWNDATA); */

   FREE(dotlist_mf);
   return py_radixdots_mf;
}


void find_matches(PyArrayObject * Py_matchlist,
		  PyArrayObject * Py_radixdots_mf, PyArrayObject * Py_cumsum_mf,
		  PyArrayObject * Py_hough, int dx, int dy, int diffspeech, int Nmax, float rhothr) {

  DotXV * radixdots_mf = (DotXV *) PyArray_DATA(Py_radixdots_mf);
  int * cumsum_mf = (int *) PyArray_DATA(Py_cumsum_mf);
  float * hough = (float *) PyArray_DATA(Py_hough);
  int compfact = diffspeech + 1;
  int dotcnt = PyArray_DIM(Py_radixdots_mf, 0);
  
   // Compute rho list
   fprintf(stderr, "Computing rholist: "); tic();
   int rhocnt = count_rholist(compfact*Nmax,hough,rhothr);
   int *rholist = (int *)MALLOC(rhocnt*sizeof(int));
   float *rhoampl = (float *)MALLOC(rhocnt*sizeof(float));
   rhocnt = compute_rholist(compfact*Nmax,hough,rhothr,rholist,rhoampl);
   fprintf(stderr, "%f s\n",toc());

   // Compute the matchlist
   fprintf(stderr, "Computing matchlist: "); tic();

   /* PyObject* op = Py_BuildValue("[(s, s), (s, s), (s, s), (s, s), (s, s), (s, s)]", "xA", "i4", "xB", "i4", "yA", "i4", "yB", "i4", "rhoampl", "f4", "score", "f4"); */
   /* PyArray_Descr* descr; */
   /* PyArray_DescrConverter(op, &descr); */
   /* Py_DECREF(op); */
   /* npy_intp dims[] = {dotcnt}; */
   /* PyArrayObject * matchlist = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, dims, descr); */
   /* Match *matchlist = (Match *)PyArray_DATA(matchlist); */
   
   Match * matchlist = (Match*) MALLOC(MAXMATCHES*sizeof(Match));
   int matchcnt = compute_matchlist( Nmax, radixdots_mf, dotcnt, cumsum_mf, rholist, rhoampl, rhocnt, dx, dy, diffspeech, matchlist );
   fprintf(stderr, "%f s\n",toc());

   int lastmc = matchcnt;
   fprintf(stderr,"    Found %d matches in first pass\n",lastmc);

   fprintf(stderr, "Filtering by first-pass duration: "); tic();
   lastmc = duration_filter(matchlist, lastmc, 0.);
   fprintf(stderr, "%f s\n",toc());
   fprintf(stderr,"    %d matches left after duration filter\n",lastmc);

   Py_matchlist->data = matchlist;
   Py_matchlist->dimensions[0] = lastmc;
   PyArray_ENABLEFLAGS(Py_matchlist, NPY_ARRAY_OWNDATA);

}


void refine_matches(PyArrayObject* Py_matchlist, PyArrayObject* Py_feats1,
		    PyArrayObject* Py_feats2, int R, float castthr,
		    float trimthr) {
  Match* matchlist = (Match *) PyArray_DATA(Py_matchlist);
  int N1 = PyArray_DIM(Py_feats1, 0);
  int N2 = PyArray_DIM(Py_feats2, 0);
  int lastmc = PyArray_DIM(Py_matchlist, 0);
  struct signature * feats1 = signature_from_feats_array(Py_feats1);
  struct signature * feats2 = signature_from_feats_array(Py_feats2);
  fprintf(stderr, "Applying second pass: "); tic();
  sig_secondpass(matchlist, lastmc, feats1, N1, feats2, N2, R, castthr, trimthr);
  fprintf(stderr, "%f s\n",toc());

  fprintf(stderr, "Filtering by second-pass duration: "); tic();
  lastmc = duration_filter(matchlist, lastmc, 0.);
  fprintf(stderr, "%f s\n",toc());
  fprintf(stderr,"    %d matches left after duration filter\n",lastmc);
  Py_matchlist->dimensions[0] = lastmc;
}


void refine_matches2(PyArrayObject* Py_matchlist, PyArrayObject* Py_feats1, PyArrayObject* Py_feats2, int R, float castthr, float trimthr, int strategy) {
  Match* matchlist = (Match *) PyArray_DATA(Py_matchlist);
  int N1 = PyArray_DIM(Py_feats1, 0);
  int N2 = PyArray_DIM(Py_feats2, 0);
  int lastmc = PyArray_DIM(Py_matchlist, 0);
  struct signature * feats1 = signature_from_feats_array(Py_feats1);
  struct signature * feats2 = signature_from_feats_array(Py_feats2);
  fprintf(stderr, "\n");
  fprintf(stderr, "Applying second pass: "); tic();
  secondpass(matchlist, lastmc, feats1, N1, feats2, N2, R, castthr, trimthr, strategy);
  fprintf(stderr, "%f s\n",toc());

  fprintf(stderr, "Filtering by second-pass duration: "); tic();
  lastmc = duration_filter(matchlist, lastmc, 0.);
  fprintf(stderr, "%f s\n",toc());
  fprintf(stderr,"    %d matches left after duration filter\n",lastmc);
  Py_matchlist->dimensions[0] = lastmc;
}
