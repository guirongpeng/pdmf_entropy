\documentclass{article}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{ctex}          % 支持中文
\usepackage{geometry}
\usepackage{algorithm}
\usepackage{algpseudocode}
\geometry{a4paper, margin=1in}

\title{Gaussian PDMF-based Fuzzy Information Measures}
\author{}
\date{}

\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}

% 修改证明结束符号为实心黑方块（更清晰美观）
\renewcommand{\qedsymbol}{$\blacksquare$}

\begin{document}

\maketitle

\section{Gaussian PDMF-based Fuzzy Information Measures}

\subsection{Fuzzy Entropy of Membership Function}

定义函数：
\begin{equation}
H(\mu) = \int_0^1 \left[ -f(x;\mu)\ln f(x;\mu) - (1-f(x;\mu))\ln(1-f(x;\mu)) \right] \, dx
\end{equation}
其中
\begin{equation}
f(x;\mu) = \Phi\bigl(\tan(\pi x - \tfrac{\pi}{2}); \mu, 1\bigr).
\end{equation}
记
\begin{equation}
H_0 = H(0) = 0.37026.
\end{equation}

\subsection{Sample-wise Fuzzy Entropy}

对每个模糊数
\[
\tilde{x}_i = \langle x_i, d_i^-, d_i^+, \mu_i^-, \mu_i^+ \rangle,
\]
定义其模糊熵（样本模糊熵）：
\begin{equation}
E_i = \frac{1}{2H_0} \left( e^{-1/d_i^-} H(\mu_i^-) + e^{-1/d_i^+} H(\mu_i^+) \right).
\end{equation}

\begin{lemma}
$0 \le E_i \le 1$。
\end{lemma}

\textbf{Proof.} 
由于对任意 $x \in [0,1]$ 有 $0 \le f(x;\mu) \le 1$，函数 $-t\ln t - (1-t)\ln(1-t)$ 在 $[0,1]$ 上非负，故 $H(\mu) \ge 0$。当 $\mu=0$ 时模糊性最大，因此 $H(\mu) \le H_0$。又因为 $0 < e^{-1/d_i^-} < 1$ 且 $0 < e^{-1/d_i^+} < 1$，可得
\[
E_i \le 1,
\]
且显然 $E_i \ge 0$。因此 $0 \le E_i \le 1$。 
\textbf{proof}

\subsection{Induced Measure}

定义信息强度 $1 - E_i$。对其归一化，得到样本$i$的权重分布：
\begin{equation}
w_i = \frac{1 - E_i}{\sum_{k=1}^n (1 - E_k)}.
\end{equation}

\begin{lemma}
$w_i \ge 0$ 且 $\sum_{i=1}^n w_i = 1$。
\end{lemma}

\begin{proof}
由 $0 \le E_i \le 1$ 知 $w_i \ge 0$ 且分母大于 0，因此 $w_i \ge 0$，且
\[
\sum_{i=1}^n w_i = 1.
\]
\end{proof}

\begin{proposition}
该分布满足：
(1) 单调性：若 $E_i \le E_j$，则 $w_i \ge w_j$；
(2) 极值性质：$E_i = 0$ 时 $w_i$ 最大，$E_i = 1$ 时 $w_i = 0$。
\end{proposition}

\begin{proof}
(1) 由 $E_i \le E_j$ 得 $1-E_i \ge 1-E_j$，故 $w_i \ge w_j$。  
(2) 若 $E_i=0$ 则 分子为1（最大）；若 $E_i=1$ 则 $w_i=0$。
\end{proof}

\textbf{Remark.}
该构造通过将熵转化为信息强度，并经归一化诱导出一个概率测度。该测度满足以下单调关系：
\begin{itemize}
\item 若 $E_i \le E_j$，则 $w_i \ge w_j$；
\item 因而，熵越小（不确定性越低），权重越大；熵越大（不确定性越高），权重越小。
\end{itemize}

\subsection{Attribute Entropy}

定义单属性熵：
\begin{equation}
E(a) = -\sum_{i=1}^n w_i^{(a)} \ln w_i^{(a)},
\end{equation}
其中 $w_i^{(a)} \ge 0$ 且 $\sum_{i=1}^n w_i^{(a)} = 1$。

\begin{proposition}
有
\[
0 \le E(a) \le \ln n.
\]
\end{proposition}

\begin{proof}
首先证明下界。函数 $f(x) = -x\ln x$ 在 $(0,1]$ 上非负，且
\[
\lim_{x \to 0^+} -x\ln x = 0,
\]
因此 $f(x) \ge 0$ 对 $x \in [0,1]$ 成立，从而
\[
E(a) = \sum_{i=1}^n (-w_i^{(a)} \ln w_i^{(a)}) \ge 0.
\]

其次证明上界。在约束条件 $\sum_{i=1}^n w_i^{(a)} = 1$ 下，函数
\[
E(a) = -\sum_{i=1}^n w_i^{(a)} \ln w_i^{(a)}
\]
在概率单纯形上达到最大值。当且仅当
\[
w_i^{(a)} = \frac{1}{n}, \quad i=1,\dots,n,
\]
时取到最大值，此时
\[
E(a) = -\sum_{i=1}^n \frac{1}{n}\ln \frac{1}{n} = \ln n.
\]

证毕。
\end{proof}



\subsection{Attribute Subset Entropy}

对于属性子集 $S$，定义样本$i$的模糊熵：
\begin{equation}
E_i^{(S)} = 1 - \prod_{a \in S} (1 - E_i^{(a)}).
\end{equation}

在定义 $E_i^{(S)}$ 后，属性子集$S$的熵定义为：

\[
E(S) = -\sum_{i=1}^n w_i^{(S)} \ln w_i^{(S)}.
\]

其中权重分布由样本级熵诱导得到：
\[
w_i^{(S)} = \frac{1 - E_i^{(S)}}{\sum_{k=1}^n (1 - E_k^{(S)})}.
\]




\begin{proposition}
若 $A \subseteq B$，则 $E_i^{(A)} \le E_i^{(B)}$。
\end{proposition}

\textbf{Proof.} 
由于 $E_i^{(a)} \in [0,1]$，故 $0 \le 1 - E_i^{(a)} \le 1$。设 $A \subseteq B$，则
\[
\prod_{a \in B} (1 - E_i^{(a)}) 
= \left( \prod_{a \in A} (1 - E_i^{(a)}) \right)
  \left( \prod_{a \in B\setminus A} (1 - E_i^{(a)}) \right).
\]
由于
\[
\prod_{a \in B\setminus A} (1 - E_i^{(a)}) \le 1,
\]
可得
\[
\prod_{a \in B} (1 - E_i^{(a)}) 
\le \prod_{a \in A} (1 - E_i^{(a)}).
\]
即
\[
1 - E_i^{(B)} \le 1 - E_i^{(A)},
\]
从而
\[
E_i^{(A)} \le E_i^{(B)}.
\]
\textbf{proof}

\textbf{Remark 1.} 
虽然 $E_i^{(A)} \le E_i^{(B)}$ 成立，但一般不能推出 $E(A) \le E(B)$，因为概率归一化会改变分布结构。


\subsection{Joint Entropy}

定义属性子集 $A$ 和 $B$ 的联合熵为
\[
E(A,B) = E(A \cup B),
\]
其中样本联合模糊熵满足
\[
E_i^{(A \cup B)} = 1 - (1 - E_i^{(A)})(1 - E_i^{(B)}),
\]
对应的权重分布为
\[
w_i^{(A,B)} = \frac{1 - E_i^{(A \cup B)}}{\sum_{k=1}^n (1 - E_k^{(A \cup B)})}.
\]
因此
\[
E(A,B) = -\sum_{i=1}^n w_i^{(A,B)} \ln w_i^{(A,B)}.
\]

\textbf{Remark 2.} 
该定义保证了联合熵的对称性 $E(A,B) = E(B,A)$，并满足递归一致性：
\[
E(A,B,C) = E(A \cup B \cup C) = E((A \cup B) \cup C)
\]

\subsection{Conditional Entropy}

定义条件熵为
\[
E(A|B) = E(A,B) - E(B).
\]


\subsection{Mutual Information}
定义互信息：
\[
I(A;B) = E(A) + E(B) - E(A,B).
\]


\subsection{Internal \& External Significance}

为刻画单个属性在属性子集中的作用，引入内在重要性与外在重要性。


设 $A$ 为属性子集，$a$ 为单个属性，定义
\begin{align}
\mathrm{Sig}_{\mathrm{in}}(a;A) 
&= E(A) - E(A \setminus \{a\}), \\
\mathrm{Sig}_{\mathrm{out}}(a;A) 
&= E(A \cup \{a\}) - E(A),
\end{align}
分别称为属性 $a$ 关于 $A$ 的 inner significance 与 outer significance。
其中 $E(\cdot)$ 表示属性子集熵（Attribute Subset Entropy）。


\textbf{Remark 1.}
inner significance 描述从子集 $A$ 中移除属性 $a$ 所引起的熵变化，反映其在当前子集中的作用程度；outer significance 描述向子集 $A$ 中加入属性 $a$ 所带来的熵变化，刻画其补充能力。

\textbf{Remark 2.}
上述两类重要性本质上对应于熵函数 $E(S)$ 下的边际贡献（marginal contribution），其中 inner significance 对应删除操作，outer significance 对应加入操作。

\textbf{Remark 3.}
内在重要性与外在重要性的符号可用于判定属性的保留或剔除策略：

\begin{itemize}
\item 若 $\mathrm{Sig}_{\mathrm{in}}(a;A) > 0$，则移除属性 $a$ 会降低权重分布的离散程度，使分布更加集中，因此，属性 $a$ 有助于增强样本之间的区分结构，应予以保留。

\item 若 $\mathrm{Sig}_{\mathrm{out}}(a;A) > 0$，说明引入属性 $a$ 会提高权重分布的离散程度，使样本间的结构更加丰富。因此，属性 $a$ 能够提供新的区分信息或补充维度，应考虑加入。
\end{itemize}

在实际应用中，通常结合阈值 $\varepsilon > 0$，以增强算法对数值波动的鲁棒性。


\begin{algorithm}
\caption{基于模糊熵与重要性的属性约简算法}
\begin{algorithmic}[1]
\State \textbf{输入：}信息系统 $(U, C)$
\State \textbf{输出：}约简属性集 $red$
\State 初始化 $red \gets \emptyset$
\State 对 $(U, C)$ 进行模糊化处理
\State 计算 $E(C)$

\Statex

\Comment{阶段1：基于外在重要性的前向选择}
\State $flag \gets \text{true}$
\While{$flag = \text{true}$}
    \State $flag \gets \text{false}$
    \ForAll{$a \in C \setminus red$}
        \State 计算 $\mathrm{Sig}_{\mathrm{out}}(a; red)$
    \EndFor
    \State $a^\ast \gets \arg\max_{a \in C \setminus red} \mathrm{Sig}_{\mathrm{out}}(a; red)$
    \If{$\mathrm{Sig}_{\mathrm{out}}(a^\ast; red) > 0$}
        \State $red \gets red \cup \{a^\ast\}$
        \State $flag \gets \text{true}$
    \EndIf
\EndWhile

\Statex

\Comment{阶段2：基于内在重要性的后向消除}
\ForAll{$a \in red$}
    \State 计算 $\mathrm{Sig}_{\mathrm{in}}(a; red)$
    \If{$\mathrm{Sig}_{\mathrm{in}}(a; red) \le 0$}
        \State $red \gets red \setminus \{a\}$
    \EndIf
\EndFor

\Statex

\Comment{阶段3：稳定性检验与自适应修正}
\While{$|E(red) - E(C)| > \delta$}
    \State $a^\ast \gets \arg\max_{a \in C \setminus red} E(red \cup \{a\})$
    \State $red \gets red \cup \{a^\ast\}$
\EndWhile

\State \Return $red$
\end{algorithmic}
\end{algorithm}

\end{document}