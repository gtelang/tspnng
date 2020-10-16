SYNOPSIS

:   Does the Euclidean TSP for a finite set of points $P$ share an edge
    with $P$'s nearest neighbor graph? [^1] Or its $k$-NNG? Or the
    Delaunay Graph? Or indeed any poly-time computable graph spanning
    the input points? We investigate this question experimentally by
    checking the validity of this conjecture for various instances in
    TSPLIB, for which the optimal solutions have been provided and for
    other synthetic data-sets (e.g. uniformly and non-uniformly
    generated points) for which we can compute optimal or near-optimal
    tours using Concorde.

DESCRIPTION

:   This question suggested itself to the author while working on the
    Horsefly problem, a generalization of the famously $NP$-hard
    Travelling Salesman Problem [^2]. One line of attack was to get at
    some kind of structure theorem by identifying a candidate set of
    good edges from which a near-optimal solution to the horsefly
    problem could be constructed. But first off, would this approach
    work for the special case of the TSP? Answering
    *"$TSP \cap NNG \stackrel{?}{=} \varnothing$"* seemed like a good
    place to start. However, all attempts at constructing examples in
    which the intersection is *empty* failed . And so did a literature
    search. The closest matching reference we found was
    [@hougardy2014edge] which *eliminates* edges that cannot be part of
    a Euclidean TSP tour on a given instance of points, based on
    checking a few simple, local geometric inequalities.

    Bill Cook, the author of Concorde[@applegate2009certification], on
    hearing about this problem from Prof. Mitchell said that, if true,
    it could be used to speed up some of the existing experimental TSP
    heuristics. [^3]

    To spur our intuition, we investigate the conjecture experimentally
    in this short report [^4] using TSPLIB and Concorde in tandem.
    TSPLIB is an online collection of medium to large size instances for
    the Euclidean, Metric and other several variants of the TSP for
    which optimal solutions have been obtained using powerful heuristics
    implemented in libraries like Concorde or Keld-Helsgaun; the
    certificate of optimality for these instances (as always!) comes
    from comparing the tour-length of the computed against a lower bound
    computed by those very heuristics.

    For starters, we investigate the following questions [^5]: for each
    symmetric 2-D Euclidean TSP instance from TSPLIB for which we have
    an optimal solution, does

    -   $TSP \cap (k{\text -})NNG \stackrel{?}{=} \varnothing$, for
        $k=1,2,\ldots$

    -   $TSP \cap \textit{Delaunay Graph} \stackrel{?}{=} \varnothing$

    -   For question 1, in the cases that the intersection is non-empty,
        what fraction (a fourth?, a fifth?)of the $n$ edges of a
        TSP-tour share its edges with the $k$-NNG does the TSP intersect
        for various values of $k$?

    -   Are there any structural patterns observed in the intersections?
        Specifically, does *at least* one edge from the intersection
        have a *vertex* incident to the convex hull? [^6] More
        generally, is this true for every layer of the onion?

    See also Appendix A for a running wishlist of questions that come
    out during discussions.

    As an aid in constructing possible counter-examples, a GUI interface
    is provided to mouse-in points and then run the Concorde heuristic
    on it.

    The Python 3.7`+` code used to generate the data and figures in this
    paper has been attached to this pdf. If you don't have a Python
    distribution please download the freely available
    [Anaconda](https://www.anaconda.com/products/individual) distro; it
    comes with most of the batteries included. You will also need to
    install a couple of other packages. See Appendix I.

    *Yalla*, what are we waiting for?! Let's go!

[^1]: In this article, we will assume the NNG to be undirected i.e.
    after constructing the nearest neighbor graph for a point-set we
    will throw away the edge directions.

[^2]: In this report by "$TSP$", we mean $TSP$-cycle and not $TSP$-path,
    although the question is still interesting for the path case. One
    reason for focusing only on the path case, is that the TSPLIB bank
    only mentions optimal cycle solutions and not optimal path
    solutions, which can be structurally quite different! Also Concorde,
    the main library used to generate any TSP solutions also outputs
    cycles.

[^3]: Note that the landmark PTAS'es for the TSP, such as those of
    Mitchell [@mitchell1999guillotine] and Arora[@arora1996polynomial],
    are too complicated to be put into code (yes, even Python!). On the
    other hand, the Concorde library uses a whole kitchen-sink of
    practical techniques such as $k$-local swaps, branch-and-bound,
    branch-and-cut to generate near-optimal (if not optimal) tours
    relatively quickly. However,it would be interesting to investigate
    the behavior of the various graphs with respect to the techniques
    used in the PTAS'es of Mitchell and Arora. Maybe we can augment them
    with the probabilistic method to prove the existence of an
    intersection??

[^4]: This report has been written as a literate program
    [@knuth1984literate; @ramsey2008noweb] to weave together the code,
    explanations and generated data into the same document. Feedback on
    the author's preliminary stab at literate programming is most
    welcome!

[^5]: Experimental answers to other questions will be barnacled onto the
    report as it grows

[^6]: This indeed seemed to be the case in all the author's failed
    attempts at a counter-example, and so we are looking for a
    proof/disproof for this special case of the conjecture
