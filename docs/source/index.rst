######################
Riptable Documentation
######################

.. toctree::
   :maxdepth: 1
   :hidden:

   Intro to Riptable <tutorial/tutorial>
   API reference <autoapi/riptable/index>

Riptable is an open source library built for high-performance data analysis. 
It's similar to Pandas by design, but it's been optimized to meet the needs of 
Riptable's core users: quantitative analysts interacting live with large volumes 
of trading data. 

Riptable is based on NumPy, so it shares many core NumPy methods for array-based 
operations. For users who work with large datasets, Riptable improves on NumPy 
and Pandas by using multi-threading and efficient memory management, much of it 
implemented at the C++ level.

.. 
    Comment: The grid and cards below are from the sphinx-design extension:
    https://sphinx-design.readthedocs.io/en/latest/
    It's also used by NumPy, and here it's paired with a copy of 
    NumPy's CSS file (/riptable/riptable/docs/source/_static/riptable.css).
    Riptable's conf.py file tells Sphinx where the css file lives. 
    Button color options: https://sphinx-design.readthedocs.io/en/latest/badges_buttons.html
    To change the image size, look for .sd-card .sd-card-img-top in riptable.css.

.. grid:: 2

    .. grid-item-card::
        :text-align: center
        :img-top: _static/index_getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        New to Riptable? Check out the Intro to Riptable, which takes you 
        through Riptable's main concepts.

        +++
      
        .. button-ref:: /tutorial/tutorial 
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:
            
            To the Intro

    .. grid-item-card::
        :text-align: center
        :img-top: /_static/index_api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide has more detailed descriptions of the 
        functions, modules, and objects included in Riptable.

        +++
        
        .. button-ref:: autoapi/riptable/index
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:
            
            To the API Reference