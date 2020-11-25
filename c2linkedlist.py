# Circular Doubly-Linked List implementation for the Dancing Links Sudoku solver

class DLNode:

    def __init__(self, val=None, prev=None, next_=None):
        self.val = val
        self.prev = prev
        self.next = next_


class ColHeaderNode(DLNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colsum = 0


class CDLList:

    def __init__(self, val=None, colheader=False):
        self.top = ColHeaderNode(val) if colheader else DLNode(val)
        self.top.prev = self.top.next = self.top

    def getNodeByVal(self, val, return_index=False):
        node = self.top
        idx = 0
        while node.val != val:
            node = node.next
            idx += 1
            if node.val == self.top.val:
                raise ValueError(f"{val} not in list")
        ret = (node, idx) if return_index else node
        
        return ret

    # def getNodeByIndex(self, index):
    #     node = self.top
    #     idx = 0
    #     while True:
    #         node = node.next
    #         idx += 1
    #         if idx == index:
    #             break
    #         if node.val == self.top.val:
    #             raise ValueError(f"{index} out of range: list has {idx} elements")

    #     return node

    def insert(self, val, node_val=None, before=False):
        if self.top.val == None:
            self.__init__(val)
        else:
            if node_val is None:
                # insert a node before the start node: at the `end` of the list
                new_node = DLNode(val, self.top.prev, self.top)
                self.top.prev.next = new_node
                self.top.prev = new_node
            else:
                node = self.getNodeByVal(node_val)
                if before:
                    new_node = DLNode(val, node.prev, node)
                    node.prev.next = new_node
                    node.prev = new_node
                else:
                    new_node = DLNode(val, node, node.next)
                    node.next.prev = new_node
                    node.next = new_node
        
        if hasattr(self.top, "colsum"):
            self.top.colsum += 1

    def remove(self, val):
        is_empty_column_list = hasattr(self.top, "colsum") and self.top.next.val == self.top.val
        if (self.top.val == None) | is_empty_column_list:
            raise ValueError("Nothing to remove")
        node, node_idx = self.getNodeByVal(val, return_index=True)
        if (node_idx == 0) and not is_empty_column_list:
            self.__init__()
            return
        node.prev.next = node.next
        node.next.prev = node.prev

        if hasattr(self.top, "colsum"):
            self.top.colsum -= 1