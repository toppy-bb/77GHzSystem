% クラスファイルの設定
\NeedsTeXFormat{pLaTeX2e}
\ProvidesClass{colloquium}[2001/10/16 v0.01 colloquium by river2470]
%\LoadClass[a4j,twocolumn,10pt]{jarticle}
\LoadClass[a4j,twocolumn,10pt]{jsarticle}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% タイトル部定義
%---------------------------------------------------------------------------
% タイトル用の新しいマクロ
\def\subject#1{\gdef\@subject{#1}}
\def\title#1{\gdef\@title{#1}}
\def\author#1{\gdef\@author{#1}}
\def\affiliation#1{\gdef\@affiliation{#1}}
\def\date#1{\gdef\@date{#1}}
%---------------------------------------------------------------------------

%---------------------------------------------------------------------------
% \maketitleを再定義
\def\maketitle{
% twocolumn環境でonecolumnにするため
\twocolumn[

\vspace{-5mm}
\@subject

\hspace{-3mm}
\fbox{\fbox{\parbox[t]{\textwidth}{
\vspace{-3mm}
\begin{flushright}
\@date
\end{flushright}
\vspace{-5mm}
\begin{center}
{\large
\@title
}
\end{center}
\vspace{-5mm}
\begin{flushright}
\@affiliation \@author
\end{flushright}
\vspace{-3mm}
}}}
]
%==========================================================================
% 本文のパラメータの定義
%--------------------------------------------------------------------------
% 改行幅
\footnotesize
}

\renewcommand{\figurename}{Fig.}
\renewcommand{\tablename}{Table.}

\setlength{\topmargin}{0mm}
\setlength{\headheight}{0mm}
\setlength{\headsep}{0mm}
\setlength{\footskip}{15mm}
\setlength{\oddsidemargin}{-7mm}
\setlength{\evensidemargin}{-7mm}
\setlength{\footskip}{5mm}
\setlength{\textheight}{230mm}
\setlength{\textwidth}{174mm}
\setlength{\columnsep}{9mm}
%\setlength{\voffset}{-10mm}

%------------------------------
% float（図表）関係
%------------------------------
% 本文ページの上部または下部に出力される浮遊要素間に挿入される
% 垂直方向のスペース（伸縮長）
\setlength{\floatsep}{.5\Cvs}

% 本文ページの上部または下部に出力される浮遊要素と本文テキストとの間に
% 挿入される垂直方向のスペース（伸縮長）
\setlength{\textfloatsep}{.5\Cvs}

% 2段組の時の\floatsep
\setlength{\dblfloatsep}{.5\Cvs}

% 2段組の時の\textfloatsep
\setlength{\dbltextfloatsep}{.5\Cvs}


%----------------------------------------------------------------------
% 引用文献
%   3つ以上連続するやつは[1]〜[3]のようにする
%----------------------------------------------------------------------
% ieicej.clsからコピー
%%## citesort.sty ##%%
%% from "citesort.sty", a little customized
\newcount\@minsofar
\newcount\@min
\newcount\@cite@temp
\def\@citex[#1]#2{%
\if@filesw \immediate \write \@auxout {\string \citation {#2}}\fi
\@tempcntb\m@ne \let\@h@ld\relax \def\@citea{}%
\@min\m@ne%
\@cite{%
  \@for \@citeb:=#2\do {\@ifundefined {b@\@citeb}%
    {\@h@ld\@citea\@tempcntb\m@ne{\bfseries ?}%
    \@warning {Citation `\@citeb ' on page \thepage \space undefined}}%
{\@minsofar\z@ \@for \@scan@cites:=#2\do {%
  \@ifundefined{b@\@scan@cites}%
    {\@cite@temp\m@ne}
    {\@cite@temp\number\csname b@\@scan@cites \endcsname \relax}%
\ifnum\@cite@temp > \@min% select the next one to list
    \ifnum\@minsofar = \z@
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\else
    \ifnum\@cite@temp < \@minsofar
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\fi\fi\fi}\@tempcnta\@min
  \ifnum\@minsofar > \z@ % some more
    \advance\@tempcnta\@ne
    \@min\@minsofar
    \ifnum\@tempcnta=\@minsofar %   Number follows previous--hold on to it
      \ifx\@h@ld\relax
        \edef \@h@ld{\@citea\csname b@\@scan@copy\endcsname}%
      \else \edef\@h@ld{\ifmmode{]〜[}\else]〜[\fi
       \csname b@\@scan@copy\endcsname}%
      \fi
    \else \@h@ld\@citea\csname b@\@scan@copy\endcsname
          \let\@h@ld\relax
  \fi % no more
\fi}%
\def\@citea{],\penalty\@highpenalty\,[}}\@h@ld}{#1}}
%% end of citesort.sty

\def\@cite#1#2{\leavevmode\unskip
  \ifnum\lastpenalty=\z@\penalty\@highpenalty\fi% highpenalty before
  \,[{\multiply\@highpenalty 3 #1%             % triple-highpenalties within
      \if@tempswa,\penalty\@highpenalty\ #2\fi % and before note.
    }]\spacefactor\@m}


%----------------------------------------------------------------------
% 表で使う太めの罫線: \Hline
%----------------------------------------------------------------------
% IEICE Trans.からのパクリ
% 未定義の時だけ定義
\makeatletter
 \@ifundefined{Hline}{\newcommand{\Hline}{\noalign{\hrule height 0.4mm}}}
\makeatother


%----------------------------------------------------------------------
% 「°（度）」
%----------------------------------------------------------------------
\def\Degree{${}^\circ$}


%----------------------------------------------------------------------
% desclab環境
%----------------------------------------------------------------------
% 使い方 
% \begin{desclab}{ラベルが一番長いやつ}
%  \item[ラベル1] ほげー
%  \item[ラベル2] ふがー
%  \item[ラベル3] ふがふが
% \end{desclab}
\def\desclablabel#1{\bf #1\hfill}
\def\desclab#1{\list{}%
 {\settowidth{\labelwidth}{\desclablabel{#1}}
 \leftmargin\labelwidth
 \advance\leftmargin\labelsep}
 \let\makelabel\desclablabel}
\let\enddesclab\endlist


%------------------------------
% 発表文献環境(publication)の定義
%
% \begin{publication}{99}
%  \addcontentsline{toc}{chapter}{\publicationname}
%  \bibitem hogehoge
% \end{publication}
%------------------------------
\def\@publabel#1{[P#1]}
\newcommand{\publicationname}{\if@english Publications\else 発表文献\fi}
\newenvironment{publication}[1]{%
  \section*{\publicationname}\@mkboth{\publicationname}{\publicationname}%
  \small%
  \list{\@publabel{\@arabic\c@enumiv}}%
  {\settowidth\labelwidth{\@publabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \@openbib@code
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy
  \clubpenalty4000
  \@clubpenalty\clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}
{\def\@noitemerr
  {\@latex@warning{Empty `publication' environment}}%
  \endlist}
