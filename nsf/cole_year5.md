---
layout: page
title: Model Constrained DNNs for Inverse Problems
permalink: /nsfcareer/year5/cole_year5
---

Last year, we explored methods to reduce the computational cost of numerically integrating stiff ODEs and PDEs. The main speed bottlenecks for integration of stiff systems are small timestep requirements for numerical solver and often the necessity of an implicit, rather than explicit, integration scheme. To resolve the expensive time requirements of integration, we designed a machine learning, surrogate model architecture which can solve stiff systems without step size requirements and can be utilized to predict all time steps in the system in parallel. The approach has the capability to approximate stiff systems to a high degree of accuracy in a fraction of the time of the original solver.

\subsubsection{Major Findings}

We evaluated our method using three stiff systems of ODEs and two stiff PDEs: the Robertson ODE \cite{robertson1966solution}, Collisional Radiative Charge States and Energy States ODEs \cite{flychk}, and the Allen-Cahn and Cahn-Hilliard PDEs \cite{montanelli2020solvingperiodicsemilinearstiff}. All of the dynamical systems considered are highly-stiff and some even have multiscale solutions, making these problems extremely challenging for traditional machine learning integration schemes, such as Neural ODE. On each test problem, our approach outperforms two state-of-the-art machine learning techniques, Neural ODE and DeepONet.

\subsubsection{Significant Results}

Table \ref{fig:err_table} shows the prediction error, in the relative L2 norm, for each machine learning approach, on each problem. It is clear that our proposed approach outperforms all other approaches in terms of accuracy. In the table, $\mathbf{OOM}$ indicates that a model could not be trained on our hardware because the memory costs were too high. Table \ref{fig:speed_table} shows the speedup achieved by our method in comparison to a traditional numerical solver for each given problem. Row one shows the total time taken by the numerical solver and our proposed approach to compute the solutions for $1000$ initial conditions, while row two shows the same for the surrogate approach. We see that our approach achieves  $\mathcal{O}(10^1)$-$\mathcal{O}(10^3)$ speedup on all problems, in terms of wall clock computation time. Finally, Figure \ref{fig:CHex} shows a side-by-side comparison between a solution to the Cahn-Hilliard PDE and the machine learning approximation of the solution, as well as giving the point-wise absolute error between the two methods. The initial condition was not seen by the network in training. The proposed model has very close agreement to the numerical solution, with a maximum absolute error on the order of $10^{-2}$.

\begin{table}[hbt!]
\caption{Relative error on test data for the three machine learning approaches}
\begin{tabular}{ |p{3cm}||p{2cm}|p{2.5cm}|p{2.5cm}|p{2cm}|p{2.5cm}|  }
 \hline
 \multicolumn{6}{|c|}{Relative Error} \\
 \hline
 Problem & ROBER & CR charge state  & Full CR  model & Allen-Cahn & Cahn-Hilliard\\
 \hline
 Neural ODE   & $1.112\cdot10^{-2}$ & $2.209\cdot10^{0}$ & $\mathbf{OOM}$ & $6.070\cdot10^{-2}$ & $8.810\cdot10^{-2}$ \\
 \hline
 DeepONet & $1.697\cdot10^{-3}$ & $2.003\cdot10^{-2}$ & $3.968\cdot10^{-2}$ & $8.589\cdot10^{-2}$ & $2.492\cdot10^{-1}$ \\
 \hline
 Proposed Approach & $\mathbf{2.80\cdot10^{-4}}$ & $\mathbf{1.981\cdot10^{-3}}$ & $\mathbf{7.486\cdot10^{-3}}$ & $\mathbf{1.072\cdot10^{-2}}$ & $\mathbf{5.090\cdot10^{-2}}$ \\
 \hline
\end{tabular}
\label{fig:err_table}

\end{table}

\begin{table}[hbt!]

\centering
\caption{Wall clock time comparison of our proposed approach with  traditional numerical solver}
\begin{tabular}{ |p{3cm}||p{2cm}|p{2.5cm}|p{2.5cm}|p{2cm}|p{2.5cm}|  }
 \hline
 \multicolumn{6}{|c|}{Speed tests in wall clock time} \\
 \hline
 Problem & ROBER & CR charge state  & Full CR model & Allen-Cahn & Cahn-Hilliard\\
 \hline
 Numerical Solver   & 6232.8s & 11034.1s & 10955.5s & 53.25s & 212.1s \\
 \hline
 Proposed Approach & 7.648s & 7.366s & 13.604s & 3.729s & 15.485s \\
 \hline
 Speedup & $\mathbf{815.0x}$ & $\mathbf{1498.0x}$ & $\mathbf{805.3x}$ & $\mathbf{14.3x}$ & $\mathbf{13.7x}$ \\
 \hline
\end{tabular}
\label{fig:speed_table}

\end{table}

\begin{figure}[h!t!b!]      
  % \begin{subfigure}[b]{\textwidth}
  \centering
  \resizebox{1.0\linewidth}{!}{\input{NSF_2025/figs/Cole/CHexample}}
  \caption{Left to right: Prediction by numerical method for Cahn-Hilliard PDE; Prediction by machine learning approach; point-wise absolute error between the two approaches. }
  % \end{subfigure} \\
  \label{fig:CHex}
\end{figure}
