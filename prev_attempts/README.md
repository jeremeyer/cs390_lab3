previous attempts to create the style transfer code, using both the skeleton keras code and creating the pytorch code from scratch

the keras code ran into numerous version conflicts and overall poor documentation/resources as to what actually was happening beneath the surface wasted a large amount of time, which is what motivated the switch to pytorch

due to inexperience with pytorch the original idea was to effectively do everything in numpy and still use the scipy implementation of the optimization function, this turned out to be extremely difficult to implement and inefficient

eventually I looked up an actual pytorch implementation using the features pytorch offers, such as the built-in matrix library and device selection
