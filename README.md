SYNOPSIS

:   Does the Euclidean TSP for a finite set of points $P$ share an edge
    with $P$'s nearest neighbor graph? [^1] Or its $k$-NNG? Or the
    Delaunay Graph? Or indeed any poly-time computable graph spanning
    the input points? We investigate this question experimentally by
    checking the validity of this conjecture for various instances in
    TSPLIB, for which the optimal solutions have been provided.

DESCRIPTION

:   This question suggested itself to the author while working on the
    Horsefly problem, itself is a generalization of the famously
    $NP$-hard Travelling Salesman Problem [^2]. One line of attack was
    to get at some kind of "structure theorem" by identifying a
    candidate set of "good" edges from which a near-optimal solution to
    the horsefly problem could be constructed. But first off, would this
    approach work for the special case of the TSP? Answering
    *"$TSP \cap NNG \stackrel{?}{=} \varnothing$"* seemed like a good
    place to start. However, all attempts at constructing
    counter-examples in which the intersection is *empty* have, thus
    far, failed. And so has a cursory literature search. Bill Cook (the
    author of Concorde) on hearing about this problem from Prof.
    Mitchell said that, if true, it could be used to speed up some of
    the existing TSP heuristics.

    To spur our intuition, we investigate the conjecture experimentally
    in this short report [^3] using TSPLIB and Concorde in tandem.
    TSPLIB is an online collection of medium to large size instances for
    the Euclidean, Metric and other several variants of the TSP for
    which optimal solutions have been obtained using powerful heuristics
    implemented in libraries like Concorde or Keld-Helsgaun; the
    certificate of optimality for these instances (as always!) comes
    from comparing the tour-length of the computed against a lower bound
    computed by those very heuristics.

    \newpage
    For starters, we investigate the following questions [^4]: for each
    symmetric 2-D Euclidean TSP instance from TSPLIB for which we have
    an optimal solution, does

    -   $TSP \cap (k{\text -})NNG \stackrel{?}{=} \varnothing$, for
        $k=1,2,\ldots$

    -   $TSP \cap \textit{Delaunay Graph} \stackrel{?}{=} \varnothing$

    -   For question 1, in the cases that the intersection is non-empty,
        what fraction (a fourth?, a third?)of the $n$ edges of a
        TSP-tour share its edges with the $k$-NNG does the TSP intersect
        for various values of $k$?

    -   Are there any structural patterns observed in the intersections?
        Specifically, does *at least* one edge of the nearest neighbor
        graph have an edge with a *vertex* incident to the convex hull?
        [^5] More generally, is this true for every layer of the
        "onion"?

    See also Appendix A for a running wishlist of questions that come
    out during discussions.

    As an aid in constructing possible counter-examples, a GUI interface
    is provided to mouse-in points and then run the Concorde heuristic
    on it.

    The Python 3.7+ code used to generate the data and figures in this
    paper has been attached to this pdf. If you don't have a Python
    distribution please download the freely available
    [Anaconda](https://www.anaconda.com/products/individual) distro,
    that comes with all the "batteries included".

    Instructions for running the code have been relegated to the
    appendix. All development and testing was done on a Linux machine;
    minimal modification (if at all!) would be needed to run it on
    Windows or Mac. In any event, the boring technical issues can be
    hashed out on Slack.

    *Yalla*, let's go!

[^1]: In this article, we will assume the NNG to be undirected i.e.
    after constructing the nearest neighbor graph for a point-set we
    will throw away the directions of the edges.

[^2]: In this report by "$TSP$", we mean $TSP$-cycle and not $TSP$-path,
    although the question is still interesting for the path case. One
    reason for focusing only on the path case, is that the TSPLIB bank
    only mentions optimal cycle solutions and not optimal path
    solutions, which can be structurally quite different! Also Concorde,
    the main library used to generate any TSP solutions also outputs
    cycles.

[^3]: This report has been written as a literate program to weave
    together the code, explanations and generated data into the same
    document. Feedback on the author's preliminary stab at literate
    programming is most welcome!

[^4]: Experimental answers to other questions will be barnacled to the
    report as it keeps (hopefully!) growing

[^5]: This indeed seemed to be the case in all the author's failed
    attempts at a counter-example, and so are looking for a
    proof/disproof for this special vase of the conjecture
