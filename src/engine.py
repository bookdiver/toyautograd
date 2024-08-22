from __future__ import annotations
import math
import numpy as np


##########################################
######## Scalar variables ################
##########################################
class Scalar:
    """A home-made scalar object that can be recorded in a computational graph,
    used for automatic differentiation.
    """

    def __init__(
        self,
        value: float,
        _children: tuple = (),
        _op: str = "",
        _label: str = "",
    ) -> None:
        """Initialization of the gradient-recorded scalar.

        Args:
            value (float): the value of the scalar.
            _children (tuple, optional): the set of children of the node that stores the current scalar. Defaults to ().
            _op (str, optional): the name of the operation that acts on the current scalar. Defaults to ''.
            _label(str, optional): the label of the variable, for nicer graph visualization.
        """
        self.value = value
        self.grad = 0.0

        self._backward = (
            lambda: None
        )  # backward tracing function, used for chaining the input's gradients with the output's
        self._children = set(_children)  # set wrapper, for efficiency
        self._op = _op
        self._label = _label

    def __repr__(self) -> str:
        return f"Scalar(value={self.value:.3f} | grad={self.grad:.3f} | label={self._label:<10})"

    # Basic operations
    def __add__(self, other: any) -> Scalar:
        """Addition operation

        Args:
            other (any): object to be added

        Returns:
            Scalar: sum output
        """
        other = other if isinstance(other, Scalar) else Scalar(value=other)
        out = Scalar(
            value=self.value + other.value,
            _children=(self, other),
            _op="+",
        )

        def _backward() -> None:
            """Backprop for addition operation g:
            f(g(x, y)) = f(x + y)
            x.grad (self) : df/dx = df/dg * dg/dx = g.grad * 1.0
            y.grad (other): df/dy = df/dg * dg/dy = g.grad * 1.0

            NOTE: the gradient accumulation "+=" is used here, because the same variable can be used in multiple operations and the gradients should be accumulated. Same for the other operations. A more mathematical expression is:
            For a multi-scalar-variable function f(g1(x), g2(x), ... , gn(n)),
            df/dx = Σ df/dgi * dgi/dx, i = 1, 2, ..., n
            where if gi(x) = gj(x), then the gradient for either gi or gj should be accumulated.
            """
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __mul__(self, other: any) -> Scalar:
        """Multiplication operation

        Args:
            other (any): object to be multiplied

        Returns:
            Scalar: product output
        """
        other = other if isinstance(other, Scalar) else Scalar(value=other)
        out = Scalar(
            value=self.value * other.value,
            _children=(self, other),
            _op="*",
        )

        def _backward() -> None:
            """Backprop for multiplication operation g:
            f(g(x, y)) = f(x * y)
            x.grad (self) : df/dx = df/dg * dg/dx = g.grad * y
            y.grad (other): df/dy = df/dg * dg/dy = g.grad * x
            """
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _backward
        return out

    def __pow__(self, other: int | float) -> Scalar:
        """Power operation

        Args:
            other (any): power of the value

        Returns:
            Scalar: power output
        """
        assert isinstance(other, (int, float)), "Power supports ONLY float or int now"
        out = Scalar(
            value=self.value**other,
            _children=(self,),
            _op=f"**{other}",
        )

        def _backward() -> None:
            """Backprop for raising power operation g:
            f(g(x, a)) = f(x**a)
            x.grad (self): df/dx = df/dg * dg/dx = g.grad * (a * x**(a-1))
            """
            self.grad += out.grad * (other * self.value ** (other - 1))

        out._backward = _backward
        return out

    def exp(self) -> Scalar:
        """Exponential operation

        Returns:
            Scalar: exponential output
        """
        expx = math.exp(self.value)
        out = Scalar(
            value=expx,
            _children=(self,),
            _op="exp",
        )

        def backward() -> None:
            """Backprop for exponential operation g:
            f(g(x)) = f(exp(x))
            x.grad (self): df/dx = df/dg * dg/dx = g.grad * exp(x)
            """
            self.grad += out.grad * expx

        out._backward = backward
        return out

    # Activation functions
    def tanh(self) -> Scalar:
        """Hyperbolic tangent operation

        Returns:
            Scalar: tanh activation output
        """
        e = math.exp(2.0 * self.value)
        out = Scalar(
            value=(e - 1.0) / (e + 1.0),
            _children=(self,),
            _op="tanh",
        )

        def backward() -> None:
            """Backprop for tanh operation g:
            f(g(x)) = f(tanh(x))
            x.grad (self): df/dx = df/dg * dg/dx = g.grad * (1 - g(x)**2)
            """
            self.grad += out.grad * (1 - out.value**2)

        out._backward = backward
        return out

    def sigmoid(self) -> Scalar:
        """Sigmoid operation

        Returns:
            Scalar: sigmoid activation output
        """
        out = Scalar(
            value=1.0 / (1.0 + math.exp(-self.value)),
            _children=(self,),
            _op="sigmoid",
        )

        def backward() -> None:
            """Backprop for sigmoid operation g:
            f(g(x)) = f(sigmoid(x))
            x.grad (self): df/dx = df/dg * dg/dx = g.grad * g(x) * (1 - g(x))
            """
            self.grad += out.grad * out.value * (1 - out.value)

        out._backward = backward
        return out

    def relu(self) -> Scalar:
        """ReLU operation

        Returns:
            Scalar: ReLU activation output
        """
        out = Scalar(
            value=max(0, self.value),
            _children=(self,),
            _op="ReLU",
        )

        def backward() -> None:
            """Backprop for ReLU operation g:
            f(g(x)) = f(ReLU(x))
            x.grad (self): df/dx = df/dg * dg/dx = g.grad * (x > 0)
            """
            self.grad += out.grad * (self.value > 0)

        out._backward = backward
        return out

    # Other operations using defined basic operations
    def __neg__(self) -> Scalar:  # -self
        return self.__mul__(-1.0)

    def __sub__(self, other: any) -> Scalar:  # self - other
        return self.__add__(-other)

    def __truediv__(self, other: any) -> Scalar:  # self / other
        return self.__mul__(other**-1)

    # Some grammar sugar for Python to deal with the reversed order of the operands
    def __radd__(self, other: any) -> Scalar:  # other + self
        return self.__add__(other)

    def __rmul__(self, other: any) -> Scalar:  # other * self
        return self.__mul__(other)

    def __rsub__(self, other: any) -> Scalar:  # other - self
        return self.__neg__().__add__(other)

    def __rtruediv__(self, other: any) -> Scalar:  # other / self
        return self.__pow__(-1).__mul__(other)

    def backward(self) -> None:
        """The actual backprop function to call, contains building the topological sort of the graph, which ensures that before a backward call, all the grads before this node has been back propagated."""
        topo = []
        visited = set()

        def build_topo(node: Scalar) -> None:
            """Building the topological sort using recursion

            Args:
                node (Scaler): current node
            """
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # initialize the output's gradient as 1.0, because d out / d out = 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()  # call local backward one by one and update .grad


#########################################################
##############  Array / Tensor variables ################
#########################################################
class Tensor:
    """A home-made tensor object that can be stored in the computational graph,
    used for automatic differentiation.
    """
    def __init__(
        self,
        value: np.ndarray,
        _children: tuple = (),
        _op: str = "",
        _label: str = "",
    ) -> None:
        """Initialization of the gradient-recorded tensor

        Args:
            value (np.ndarray): the value of the tensor, implemented in np.ndarray
            _children (tuple, optional): the set of children of the node that stores the current tensor. Defaults to ().
            _op (str, optional): the name of the operation that acts on teh current tensor. Defaults to "".
            _label (str, optional): the label of the variable, for nicer graph visualization. Defaults to "".
        """
        self.value = value
        self.grad = np.zeros_like(value)
        
        self._backward = (
            lambda: None
        )
        self._children = set(_children)
        self._op = _op
        self._label = _label

    def __repr__(self) -> str:
        return f"Tensor(value={self.value} | grad={self.grad} | label={self.label:<10})"
    
    def __add__(self, other: any) -> Tensor:
        other = self._align_tensor(other)
        out = Tensor(
            value=self.value + other.value,
            _children=(self, other),
            _op="+",
        )
        
        def _backward() -> None:
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
            
        out._backward = _backward
        return out
    
    def __mul__(self, other: any) -> Tensor:
        """Element-wise production of two tensors
        """
        other = self._align_tensor(other)
        out = Tensor(
            value=self.value*other.value,
            _children=(self, other),
            _op="*",
        )
        
        def _backward() -> None:
            # !!! NOT SURE IF IT'S CORRECT
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: any) -> Tensor:
        """Matrix multiplication
        """
        other = self._align_tensor(other)
        out = Tensor(
            value=np.dot(self.value, other.value),
            _children=(self, other),
            _op="@"
        )
        
        def _backward() -> None:
            self.grad += np.outer(out.grad, other.value)
            other.grad += np.dot(self.value.T, out.grad)
            
        out._backward = _backward
        return out
    
    def mean(self) -> Tensor:
        """Mean operation, used for obtaining scalar output

        Returns:
            Tensor: mean of all the entries in the tensor.
        """
        out = Tensor(
            value=np.mean(self.value),
            _children=(self, ),
            _op="mean",
        )
        
        def _backward() -> None:
            """Backprop for mean operation f:
            f(x) = (Σ xi) / n
            df/dxi = 1 / n
            df/dx = [1/n, 1/n, ... , 1/n]
            """
            n = self.value.shape
            self.grad += np.ones_like(self.value) / n
        
        out._backward = _backward
        return out
    
    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(node: Scalar) -> None:
            """Building the topological sort using recursion

            Args:
                node (Scaler): current node
            """
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # initialize the output's gradient as 1.0, because d out / d out = 1.0
        self.grad = np.ones_like(self.value)
        for node in reversed(topo):
            node._backward()  # call local backward one by one and update .grad
    
    def _align_tensor(self, other: any) -> Tensor:
        if isinstance(other, Tensor):
            other = other
        elif isinstance(other, float):
            other = Tensor(value=other * np.ones_like(self.value))
        elif isinstance(other, np.ndarray):
            assert other.shape[-1] == self.value.shape[0], f"Shape does not match, receiving {other.shape} and {self.value.shape}."
            other = Tensor(value=other)
        else:
            raise ValueError(f"operation object must be float, np.ndarray or Tensor, but receiving {type(other)}.")
        
        return other


#########################################################
######### Computational Graph Visualization #############
#########################################################


def draw_graph(root: Scalar, format: str = "svg", rankdir="LR"):
    """Draw the computational graph

    Args:
        root (Scalar): the final output variable
        format (str, optional): visualization format. Defaults to 'svg'.
        rankdir (str, optional): plotting direction. Defaults to 'LR'.

    Returns:
        graph (_type_): _description_
    """
    from graphviz import Digraph

    def trace(root: Scalar) -> tuple:
        """DFS traverse the computational graph

        Args:
            root (Scalar): the final output variable

        Returns:
            tuple: node and edge collections
        """
        nodes, edges = set(), set()

        def build_block(node: Scalar) -> None:
            if node not in nodes:
                nodes.add(node)
                for child in node._children:
                    edges.add((child, node))
                    build_block(child)

        build_block(root)
        return nodes, edges

    assert (
        rankdir in ["LR", "TB"]
    ), f"rankdir must be either LR (Left-to-Right) or TB (Top-to-Bottom), but received {rankdir}."
    nodes, edges = trace(root)
    graph = Digraph(
        format=format, graph_attr={"rankdir": rankdir}, node_attr={"rankdir": rankdir}
    )

    for node in nodes:
        graph.node(
            name=str(id(node)),
            label=f"{node._label} | value: {node.value:.3f} | grad: {node.grad:.3f}",
            shape="record",
        )
        if node._op:
            graph.node(name=str(id(node)) + node._op, label=node._op)
            graph.edge(str(id(node)) + node._op, str(id(node)))

    for node1, node2 in edges:
        graph.edge(str(id(node1)), str(id(node2)) + node2._op)

    return graph
