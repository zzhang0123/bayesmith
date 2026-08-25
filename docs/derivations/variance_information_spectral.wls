#!/usr/bin/env wolframscript

(* Is there ONE form covering both rows the gate accepts?

   Claim: whenever N(theta) is diagonalised by a theta-INDEPENDENT basis,

       1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 Sum_k d_a log lambda_k d_b log lambda_k

   Both accepted rows have a fixed basis: a diagonal Normal has U = I with
   lambda_i = sigma_i^2, a CirculantNormal has U = DFT whatever its kernel
   does. If so, the shipped 2 (dlog sigma/dx)^T (dlog sigma/dx) is the
   SPECIAL CASE lambda_i = sigma_i^2 rather than a different rule, and the
   correlated form is a generalisation rather than a patch.

   The claim predicts its own failure: a covariance whose EIGENBASIS moves
   with theta is not covered. That is tested too. *)

circulantEigen[col_, nn_] :=
  Table[Sum[col[[1 + j]] Cos[2 Pi j k/nn], {j, 0, nn - 1}], {k, 0, nn - 1}];

Print["=========================================================="];
Print["(1) the shipped diagonal formula IS the spectral formula"];
Print["=========================================================="];
Module[{nn = 3, sig, NN, exact, spectral, shipped, pars},
  sig = Table[Symbol["s" <> ToString[i]][a, b], {i, nn}];
  NN = DiagonalMatrix[sig^2]; pars = {a, b};
  shipped = Table[Simplify[2 Sum[D[Log[sig[[i]]], pars[[p]]] D[Log[sig[[i]]], pars[[q]]], {i, nn}]], {p, 2}, {q, 2}];
  spectral = Table[Simplify[1/2 Sum[D[Log[sig[[i]]^2], pars[[p]]] D[Log[sig[[i]]^2], pars[[q]]], {i, nn}]], {p, 2}, {q, 2}];
  exact = Table[Simplify[1/2 Tr[Inverse[NN] . D[NN, pars[[p]]] . Inverse[NN] . D[NN, pars[[q]]]]], {p, 2}, {q, 2}];
  Print["  shipped  == spectral : ", Simplify[shipped - spectral] === ConstantArray[0, {2, 2}]];
  Print["  spectral == exact tr : ", Simplify[spectral - exact] === ConstantArray[0, {2, 2}]];
];

Print[""];
Print["=========================================================="];
Print["(2) a CIRCULANT whose kernel changes SHAPE with theta"];
Print["=========================================================="];
Do[Module[{nn = m, col, Cc, lam, exact, spectral},
   (* every free entry its own function of theta: the spectrum moves and so
      does the correlation SHAPE, not merely an overall scale *)
   col = Table[Symbol["g" <> ToString[Min[i, nn - i]]][th], {i, 0, nn - 1}];
   Cc = Table[col[[1 + Mod[j - i, nn]]], {i, 0, nn - 1}, {j, 0, nn - 1}];
   lam = circulantEigen[col, nn];
   exact = Simplify[1/2 Tr[Inverse[Cc] . D[Cc, th] . Inverse[Cc] . D[Cc, th]]];
   spectral = Simplify[1/2 Sum[D[Log[lam[[k]]], th]^2, {k, nn}]];
   Print["  n=", nn, " : exact - spectral simplifies to ", Simplify[exact - spectral]];
   ], {m, {3, 4, 5}}];

Print[""];
Print["=========================================================="];
Print["(3) the predicted FAILURE: an eigenbasis that moves"];
Print["=========================================================="];
Module[{mu, Cm, NN, exact, lam, spectral, U1, U2},
  mu = {a + b, a + 2 b, a + 5 b};      (* per-sample D, not a common scale *)
  Cm = {{1, 1/3, 1/9}, {1/3, 1, 1/3}, {1/9, 1/3, 1}};
  NN = DiagonalMatrix[mu] . Cm . DiagonalMatrix[mu];
  U1 = Chop[N[Eigenvectors[NN /. {a -> 1, b -> 1}]]];
  U2 = Chop[N[Eigenvectors[NN /. {a -> 1, b -> 2}]]];
  Print["  eigenbasis fixed in theta? ", Chop[U1 - U2] === ConstantArray[0, {3, 3}]];
  exact = 1/2 Tr[Inverse[NN] . D[NN, b] . Inverse[NN] . D[NN, b]] /. {a -> 1, b -> 1};
  lam = Eigenvalues[N[NN]];
  spectral = 1/2 Total[(D[lam, b]/lam)^2] /. {a -> 1, b -> 1};
  Print["  exact    = ", N[exact]];
  Print["  spectral = ", N[spectral]];
  Print["  agree?     ", Chop[N[exact] - N[spectral]] === 0];
];

Print[""];
Print["=========================================================="];
Print["(4) and the same failure, stated as what IS safe"];
Print["=========================================================="];
Module[{Cm, NN, exact, lam, spectral},
  Cm = {{1, 1/3, 1/9}, {1/3, 1, 1/3}, {1/9, 1/3, 1}};
  NN = c[th]^2 Cm;                       (* a COMMON scale: basis fixed *)
  exact = Simplify[1/2 Tr[Inverse[NN] . D[NN, th] . Inverse[NN] . D[NN, th]]];
  lam = c[th]^2 Eigenvalues[Cm];
  spectral = Simplify[1/2 Total[(D[lam, th]/lam)^2]];
  Print["  N = c(theta)^2 C  ->  exact = ", exact, " ; spectral = ", spectral];
  Print["  agree? ", Simplify[exact - spectral] === 0];
  Print["  and the SHIPPED formula, 2 Sum_i dlog(sigma_i)^2 with"];
  Print["  sigma_i = c(theta) sqrt(C_ii):  ",
   Simplify[2 Sum[D[Log[c[th] Sqrt[Cm[[i, i]]]], th]^2, {i, 3}]]];
];
