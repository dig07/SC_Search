# SC_Search

Python library for semi-coherent searches of stellar mass binary signals in LISA. 
![Tracks](https://github.com/dig07/SC_Search/assets/23508858/c4dd3efb-16b5-46f6-b3cf-6e672ceacd5c)

<hr>
<b>Please note this library is still in a very hacky state and under development! A lot of the functionality such as the waveform model choices right now is hardcoded, however the bits of the code relevant to the semi-coherent method should be relatively easy to read and re-implement from the file: https://github.com/dig07/SC_Search/blob/main/SC_search/Semi_Coherent_Functions.py.</b>

<hr>
If you want to install this library in addition to the following packages, you also need an environment with a working installation of <a href="https://cupy.dev/">cupy</a> which allows you to run the search (since it needs GPUs). <br><br>
Install order (using a conda environment): 
<ul>
  <li>BBHx, used for response and interpolation (https://github.com/mikekatz04/BBHx, following the install instructions there).</li>
  <li>Optionally install Balrog if you have access to it, only used to verify the responses are consistent between the BBHx and Balrog codes, not used in the actual search.</li>
  <li>SC_Search (This package)</li>
  <li><a href="https://github.com/dig07/PySO/tree/main/PySO">PySO</a> (Particle swarm optimisation library)</li>
</ul>
