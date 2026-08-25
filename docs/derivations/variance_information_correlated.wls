#!/usr/bin/env wolframscript
(* ::Package:: *)

(* The correlated form of the variance-information term.

   For d ~ N(mu(theta), N(theta)) the Fisher matrix is

       F_ab = (d_a mu)^T N^-1 (d_b mu) + 1/2 tr(N^-1 d_a N  N^-1 d_b N)

   exact/fisher.py applies the SECOND term as 2 (dlog sigma/dx)^T
   (dlog sigma/dx), derived for a DIAGONAL N, and under a radiometer that
   collapses to a clean factor (1 + 2 f^2).  Nothing said it survives
   off-diagonal terms, so `precision_parts` refuses a correlated node that
   also claims depends_on_prediction=True.

   THE MODEL, written down before differentiating, because a different
   reading gives a different term:

       N(theta) = D(theta) . C . D(theta),   D = DiagonalMatrix[f mu_i(theta)]

   -- multiplicative noise with a FIXED correlation structure C.  C is a
   correlation matrix (unit diagonal); all the parameter dependence is in D.

   Everything below is checked symbolically at general n and then at
   explicit small n.  The partial hand-derivation this is testing:

       G_a = D^-1 d_a D    =>    N^-1 d_a N = D^-1 C^-1 G_a C D + G_a

   and, for a single scaling parameter (mu = theta x, so G_a = I/theta),
   that collapsing to 2 I/theta, giving the factor

       1 + 2 f^2 n / (1^T C^-1 1)

   which reduces to 1 + 2 f^2 at C = I, and for a circulant C has
   1^T C^-1 1 = n/lambda0.
*)

Print["=============================================================="];
Print["(1) the general identity  N^-1 d_a N = D^-1 C^-1 G_a C D + G_a"];
Print["=============================================================="];

(* general symbolic n=3: an arbitrary symmetric C and an arbitrary
   theta-dependent diagonal D.  Nothing about mu is assumed yet. *)
n = 3;
CC = Table[If[i == j, 1, Symbol["c" <> ToString[Min[i, j]] <> ToString[Max[i, j]]]],
   {i, n}, {j, n}];
dvec = Table[Symbol["d" <> ToString[i]][th], {i, n}];
DD = DiagonalMatrix[dvec];
NN = DD . CC . DD;

dN = D[NN, th];
Ninv = Inverse[NN];
lhs = Ninv . dN;

GG = Inverse[DD] . D[DD, th];
rhs = Inverse[DD] . Inverse[CC] . GG . CC . DD + GG;

identity = Simplify[lhs - rhs];
Print["  N^-1 d_a N - (D^-1 C^-1 G_a C D + G_a) = ", MatrixForm[identity]];
Print["  identity holds: ", identity === ConstantArray[0, {n, n}] ||
   Simplify[Total[Flatten[identity]^2]] === 0];

Print[""];
Print["=============================================================="];
Print["(2) a single SCALING parameter: mu = theta x, so D = f theta diag(x)"];
Print["=============================================================="];

Clear[n];
Do[
  Module[{nn = m, x, Dm, Nm, dNm, NidN, second, first, J, ratio, Cm},
   x = Table[Symbol["x" <> ToString[i]], {i, nn}];
   Cm = Table[
     If[i == j, 1, Symbol["c" <> ToString[Min[i, j]] <> ToString[Max[i, j]]]],
     {i, nn}, {j, nn}];
   Dm = f th DiagonalMatrix[x];
   Nm = Dm . Cm . Dm;
   dNm = D[Nm, th];
   NidN = Simplify[Inverse[Nm] . dNm];
   Print["  n=", nn, ": N^-1 d_th N = ", MatrixForm[Simplify[NidN]]];
   second = Simplify[1/2 Tr[NidN . NidN]];
   J = D[th x, th];
   first = Simplify[J . Inverse[Nm] . J];
   ratio = Simplify[second/first];
   Print["    1/2 tr(...)          = ", second];
   Print["    J^T N^-1 J           = ", Simplify[first]];
   Print["    ratio                = ", Simplify[ratio]];
   Print["    ratio vs 2 f^2 n / (1.C^-1.1) : ",
    Simplify[ratio - 2 f^2 nn/(ConstantArray[1, nn] . Inverse[Cm] . ConstantArray[1, nn])]];
   ], {m, {2, 3, 4}}];

Print[""];
Print["=============================================================="];
Print["(3) the two reductions the hand derivation claims"];
Print["=============================================================="];

Do[Module[{nn = m, Cid, one},
   one = ConstantArray[1, nn];
   Cid = IdentityMatrix[nn];
   Print["  n=", nn, "  C=I   : factor = ",
    Simplify[1 + 2 f^2 nn/(one . Inverse[Cid] . one)], "   (want 1 + 2 f^2)"];
   ], {m, {2, 3, 4, 5}}];

Print[""];
Print["  circulant C: is 1^T C^-1 1 = n/lambda0, lambda0 = Total[first column]?"];
Do[Module[{nn = m, col, Cc, one, lhsq, rhsq},
   col = Table[Symbol["k" <> ToString[i]], {i, 0, nn - 1}];
   (* symmetric circulant: k_i = k_{n-i} *)
   col = Table[col[[1 + Min[i, nn - i]]], {i, 0, nn - 1}];
   Cc = Table[col[[1 + Mod[j - i, nn]]], {i, 0, nn - 1}, {j, 0, nn - 1}];
   one = ConstantArray[1, nn];
   lhsq = Simplify[one . Inverse[Cc] . one];
   rhsq = Simplify[nn/Total[col]];
   Print["    n=", nn, " : 1^T C^-1 1 = ", lhsq, " ; n/lambda0 = ", rhsq,
    " ; equal: ", Simplify[lhsq - rhsq] === 0];
   ], {m, {3, 4, 5, 6}}];

Print[""];
Print["=============================================================="];
Print["(4) so, for a CIRCULANT correlation, the factor is..."];
Print["=============================================================="];
Print["  1 + 2 f^2 n / (n/lambda0) = 1 + 2 f^2 lambda0,"];
Print["  lambda0 = Total[first column of C] = C's DC eigenvalue."];
Print["  At C = I, lambda0 = 1 and this is 1 + 2 f^2, as it must be."];

Print[""];
Print["=============================================================="];
Print["(5) does the collapse to a SCALAR factor survive >1 parameter?"];
Print["=============================================================="];
Module[{nn = 3, x, y, Dm, Nm, Fvar, Fmean, mu, pars},
  x = Table[Symbol["x" <> ToString[i]], {i, nn}];
  y = Table[Symbol["y" <> ToString[i]], {i, nn}];
  mu = a x + b y;
  Dm = f DiagonalMatrix[mu];
  Nm = Dm . IdentityMatrix[nn] . Dm;
  pars = {a, b};
  Fvar = Table[
    Simplify[1/2 Tr[Inverse[Nm] . D[Nm, pars[[p]]] . Inverse[Nm] . D[Nm, pars[[q]]]]],
    {p, 2}, {q, 2}];
  Fmean = Table[
    Simplify[D[mu, pars[[p]]] . Inverse[Nm] . D[mu, pars[[q]]]], {p, 2}, {q, 2}];
  Print["  F_var  = ", MatrixForm[Fvar]];
  Print["  F_mean = ", MatrixForm[Fmean]];
  Print["  is F_var == 2 f^2 F_mean (i.e. a scalar factor)? ",
   Simplify[Fvar - 2 f^2 Fmean] === ConstantArray[0, {2, 2}]];
];
