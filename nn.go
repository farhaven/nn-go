package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
)

type Node interface {
	getValue() float64
	updateValue()
}

type ConstantNode struct {
	value float64
}

func (c *ConstantNode) String() string {
	return fmt.Sprintf(`C:%p:%f`, c, c.value)
}

func (c *ConstantNode) getValue() float64 {
	return c.value
}

func (c *ConstantNode) updateValue() {
	/* Do nothing here, the value does not depend on any other nodes */
}

type SumNode struct {
	f      func(float64) float64
	inputs []Node
	value  float64
}

func (s *SumNode) updateValue() {
	sum := float64(0)

	for _, input := range s.inputs {
		sum += input.getValue()
	}

	s.value = s.f(sum)
}

func (s *SumNode) getValue() float64 {
	return s.value
}

type Network struct {
	nodes      []Node
	numInputs  int
	numOutputs int
}

func NewNetwork(numInputs, numOutputs int) Network {
	nodes := []Node{}
	for idx := 0; idx < numInputs; idx++ {
		nodes = append(nodes, &ConstantNode{})
	}

	for idx := 0; idx < numOutputs; idx++ {
		nodes = append(nodes, &SumNode{
			f:      math.Tanh,
			inputs: nodes[:numInputs],
		})
	}

	return Network{
		nodes:      nodes,
		numInputs:  numInputs,
		numOutputs: numOutputs,
	}
}

func (n *Network) feed(inputs []float64) {
	if len(inputs) != n.numInputs {
		panic(`invalid input length`)
	}

	/* Set values for input layer */
	for idx, input := range inputs {
		node := n.nodes[idx].(*ConstantNode)
		node.value = input
	}

	/* Iterate over remaining layers and update node values */
	for _, node := range n.nodes[n.numInputs:] {
		node.updateValue()
	}
}

/* Adds a random edge between two nodes. */
func (n *Network) addRandomEdge() {
	srcIdx := rand.Intn(len(n.nodes) - n.numOutputs)
	dstIdx := 0
	for {
		dstIdx = rand.Intn(len(n.nodes)-n.numInputs) + n.numInputs
		if srcIdx < dstIdx {
			break
		}
	}

	srcNode := n.nodes[srcIdx]
	dstNode := n.nodes[dstIdx].(*SumNode)

	dstNode.inputs = append(dstNode.inputs, srcNode)
}

/* Splits a random edge between two nodes. */
func (n *Network) splitRandomEdge() {
	dstIdx := rand.Intn(len(n.nodes) - n.numInputs) + n.numInputs
	dstNode := n.nodes[dstIdx].(*SumNode)

	log.Println(`dest idx:`, dstIdx, `dest node:`, dstNode)

	srcNode := dstNode.inputs[rand.Intn(len(dstNode.inputs))]
	srcIdx := -1
	for idx, node := range n.nodes[:dstIdx] {
		if node == srcNode {
			srcIdx = idx
			break
		}
	}
	if srcIdx == -1 {
		panic(`can't find index for source node`)
	}

	log.Println(`src idx:`, srcIdx, `src node:`, srcNode)

	/* Create new middle node */
	middleNode := &SumNode{
		f: math.Tanh,
		inputs: []Node{srcNode},
	}

	/* Replace src with middle node in dst's input list */
	newInputs := []Node{}
	for _, input := range dstNode.inputs {
		if input != srcNode {
			newInputs = append(newInputs, input)
		}
	}
	newInputs = append(newInputs, middleNode)
	dstNode.inputs = newInputs

	/* Insert middle not right after src, or after last input if src is an input node */
	middleIdx := srcIdx + 1
	if srcIdx < n.numInputs {
		middleIdx = n.numInputs
	}
	n.nodes = append(n.nodes, nil)
	copy(n.nodes[middleIdx+1:], n.nodes[middleIdx:])
	n.nodes[middleIdx] = middleNode
}

func (n *Network) getOutput() []float64 {
	res := []float64{}

	for _, node := range n.nodes[len(n.nodes)-n.numOutputs:] {
		res = append(res, node.getValue())
	}

	return res
}

func (n *Network) dumpDot() {
	fh, err := os.Create(`net.dot`)
	if err != nil {
		log.Fatalln(`can't create file:`, err)
	}
	defer fh.Close()

	fh.WriteString("digraph {\n")
	for _, node := range n.nodes {
		if sn, ok := node.(*SumNode); ok {
			fh.WriteString(fmt.Sprintf("\t\"%p\" [label=\"%f\"]", sn, sn.value))
			for _, input := range sn.inputs {
				fh.WriteString(fmt.Sprintf("\t\"%p\" -> \"%p\";\n", input, sn))
			}
		} else {
			cn := node.(*ConstantNode)
			fh.WriteString(fmt.Sprintf("\t\"%p\" [style=filled, fillcolor=green, label=\"%f\"]\n", cn, cn.value))
		}
	}
	fh.WriteString("}\n")

	log.Println(`network written to net.dot`)
}

func main() {
	log.Println(`here we go`)

	net := NewNetwork(2, 1)

	log.Println(`network:`, net)

	net.feed([]float64{1, 0})
	log.Println(`network output:`, net.getOutput())

	net.splitRandomEdge()
	net.addRandomEdge()
	net.splitRandomEdge()
	net.addRandomEdge()
	net.splitRandomEdge()
	net.addRandomEdge()
	net.splitRandomEdge()

	net.dumpDot()

	net.feed([]float64{1, 0})
	log.Println(`network output:`, net.getOutput())
}
