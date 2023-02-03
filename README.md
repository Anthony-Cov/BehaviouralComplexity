For the paper 
<H1>Predictability of Consumer Behaviour Regarding to Its Complexity</H1>
<p> <b>Abstract</b>. For ages analytics were dealing with the data which repre-
sent the financial processes of large groups of agents and the whole pop-
ulation. Nowaday technologies that help to handle big data allow us to
come closer to every customer and explore his personal behaviour. Here
we investigate the consumer behaviour predictability using transactions
which satisfy different basic customer values. We use the methods of data
compression to evaluate the complexity of customer’s activity. We show
that Kolmogorov complexity helps to estimate behavioural predictability
of every single customer which can be useful in data preprocessing for
training some forecasting models dealing with the whole population, or
in some problems of targeting and marketing.
</p>
<H2>Content</H2>
<ul>
<li><b>nabiim.py</b> - predictability-measuring neural network model, Huffman and LZW compression procedures, LZ-complexity calculation, and some utilites for import.
<li><b>pr19.py</b> - neural-network forecasting model for a single customer predictability measuring.
<li><b>raif_values.zip</b> - archive with raif_values.csv Data set with transactions of Raiffeisen bank preprocessed for methods inplementation.
<li><b>cmpl_19.csv</b> - the results of realized predictability estimation and complexity calculations for private data set of 5100 customers.<br>
<ul>
<i>id</i> - customer identifier,<br>
<i>survival, socialization, self_realization, aggregated</i> - hit probability on basic values and aggregated predictability for every single customer,<br>
<i>Huffman, LZW</i> - behaviour complexity by Huffman and Lempel-Zov-Welch algorithms,<br>
<i>LZ_survival, LZ_socialization, LZ_self_realization</i> - Lempel-Ziv complexity for basic values separately
</ul>
</ul>
<li><b>ComplPredic.ipynb</b> - experiments for the research.
