# Patient and Protocol Representation
Patients and protocols are represented as vectors within a shared vector space, initially seven-dimensional, corresponding to MoCA cognitive subscales.

## Patient Embeddings
Patient vector calculated as follows:

1. For MoCA subscales, deficiency \(d_i\):

   $
   d_i = S^i_{\max} - s_i
   $

2. Deficiency vector:

   $
   d = [d_1, d_2, \dots, d_7]
   $

3. Normalize to form patient embedding \(p\):

   $
   p = \frac{d}{\sum_{i=1}^{7} d_i}
   $

## Protocol Embeddings
Clinician-rated protocol vectors:

1. Rating vector provided by clinicians:

   $
   r = [r_1, r_2, \dots, r_7]
   $

2. Normalized protocol embedding \(q\):

   $
   q = \frac{r}{\sum_{i=1}^{7} r_i}
   $

## Patient Profile Update
Profile updates at assessment intervals (T1, T2 every 4 weeks):

- Assessments at T0 (initial), T1 (week 4), T2 (week 8), T3 (final clinical assessment).
- Diagnostic protocol embedding \(q\), success rate \(s_T\) gives weighted embedding:

  $
  e_T = s_T \cdot q
  $

- Update estimated MoCA using embedding ratio between times T0, T1:

  $
  \frac{\text{MoCA}_0}{\text{MoCA}_1} = \frac{\sum_i e_{0,i}}{\sum_i e_{1,i}}
  $

Rearranged as:

  $
  \text{MoCA}_1 = \text{MoCA}_0 \times \frac{\sum_i e_{1,i}}{\sum_i e_{0,i}}
  $

Deficiencies recalculated and embeddings updated accordingly.