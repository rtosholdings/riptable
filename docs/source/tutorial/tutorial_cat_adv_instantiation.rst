
A Useful Way to Instantiate a Categorical
*****************************************

It can sometimes be useful to instantiate a Categorical with only one
category, then fill it in as needed.

For example, let’s say we have a Dataset with a column that has a lot of
categories, and we want to create a new Categorical column that keeps
two of those categories, properly aligned with the rest of the data in
the Dataset, and lumps the other categories into a category called
‘Other.’

Our Dataset, with a column of many categories::

    >>> rng = np.random.default_rng(seed=42)
    >>> N = 50
    >>> ds_buildcat = rt.Dataset({'big_cat': rng.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], N)})
    >>> ds_buildcat
      #   big_cat
    ---   -------
      0   D      
      1   I      
      2   A      
      3   I      
      4   F      
      5   B      
      6   D      
      7   F      
      8   D      
      9   B      
     10   G      
     11   G      
     12   B      
     13   C      
     14   C      
    ...   ...    
     35   I      
     36   J      
     37   D      
     38   C      
     39   J      
     40   G      
     41   C      
     42   G      
     43   F      
     44   J      
     45   C      
     46   J      
     47   J      
     48   B      
     49   B    

We create our ‘small’ Categorical instantiated with 3s, which fills the
column with the ‘Other’ category::

    >>> ds_buildcat.small_cat = rt.Cat(rt.full(ds_buildcat.shape[0], 3), categories=['B', 'D', 'Other']) 
    >>> ds_buildcat.small_cat
    >>> ds_buildcat
      #   big_cat   small_cat
    ---   -------   ---------
      0   D         Other    
      1   I         Other    
      2   A         Other    
      3   I         Other    
      4   F         Other    
      5   B         Other    
      6   D         Other    
      7   F         Other    
      8   D         Other    
      9   B         Other    
     10   G         Other    
     11   G         Other    
     12   B         Other    
     13   C         Other    
     14   C         Other    
    ...   ...       ...      
     35   I         Other    
     36   J         Other    
     37   D         Other    
     38   C         Other    
     39   J         Other    
     40   G         Other    
     41   C         Other    
     42   G         Other    
     43   F         Other    
     44   J         Other    
     45   C         Other    
     46   J         Other    
     47   J         Other    
     48   B         Other    
     49   B         Other  

Now we can fill in the aligned ‘B’ and ‘D’ categories::

    >>> ds_buildcat.small_cat[ds_buildcat.big_cat == 'B'] = 'B'
    >>> ds_buildcat.small_cat[ds_buildcat.big_cat == 'D'] = 'D'
    >>> ds_buildcat
      #   big_cat   small_cat
    ---   -------   ---------
      0   D         D        
      1   I         Other    
      2   A         Other    
      3   I         Other    
      4   F         Other    
      5   B         B        
      6   D         D        
      7   F         Other    
      8   D         D        
      9   B         B        
      10  G         Other    
      11  G         Other    
      12  B         B        
      13  C         Other    
      14  C         Other    
     ...  ...       ...      
      35  I         Other    
      36  J         Other    
      37  D         D        
      38  C         Other    
      39  J         Other    
      40  G         Other    
      41  C         Other    
      42  G         Other    
      43  F         Other    
      44  J         Other    
      45  C         Other    
      46  J         Other    
      47  J         Other    
      48  B         B        
      49  B         B  