class Positions:
    """Access positions around a node in a context.
    
    For a given origin node, provides data on another node
    that is (+/-)N positions away in a context.
    """
    
    def __init__(self, element, positions, default=None):
        """Prepare context and positions for a supplied TF node.
        
        Arguments:
            n: an integer that is a Text-Fabric node that serves
                as the origin node.
            context: a string that specifies a context in which
                to search for a position relative to origin node.
            tf: an instance of Text-Fabric with a loaded corpus.
            order: The method of order to use for the search.
                Options are "slot" or "node."
        """
    
        # set up elements and positions
        self.element = element
        self.positions = positions
        self.originindex = self.positions.index(element)
        self.default = default
    
    def elementpos(self, position):
        """Get position using order of context.
        
        !CAUTION!
            This method should only be used with
            linguistic units known to be non-overlapping. 
            A TF "slot" is a good example for which this 
            method might be used. 
            
            For example, given a phrase with another embedded phrase:
            
                > [1, 2, 3, [4, 5], 6]
                
            this method will indicate that slots 3 and 6 are adjacent
            with respect to the context. This is OK because we know 
            3 and 6 do not embed one another.
            By contrast, TF locality methods would mark 3 and 4 as adjacent.
            
        Arguments: 
            position: integer that is the position to find
                from the origin element
        """
        # use index in positions to get adjacent node
        # return None when exceeding bounds of context
        pos_index = self.originindex + position
        pos_index = pos_index if (pos_index > -1) else None
        try:
            return self.positions[pos_index]
        except (IndexError, TypeError):
            return None

    def get(self, position, default=None, do=None):
        """Get data on node (+/-)N positions away. 
        
        Arguments:
            position: a positive or negative integer that 
                tells how far away the target node is from source.
            returndata: a function that should be called and returned 
                in case of a match.
        """

        # get global default
        default = default if default is not None else self.default

        # get requested position in context
        get_pos = self.elementpos(position)

        # return requested data
        if get_pos:
            if not do:
                return get_pos
            else:
                return do(get_pos)

        # return empty data
        else:
            return default
