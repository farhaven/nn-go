package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
)

type Network struct {
	nodes      []Node
	numInputs  int
	numOutputs int
	totalError float64
}

func NewNetwork(numInputs, numOutputs int) *Network {
	numInputs += 1 /* Add a bias node */
	nodes := []Node{}
	for idx := 0; idx < numInputs; idx++ {
		nodes = append(nodes, &ConstantNode{1})
	}

	for idx := 0; idx < numOutputs; idx++ {
		nodes = append(nodes, NewSumNode(nodes[:numInputs]))
	}

	return &Network{
		nodes:      nodes,
		numInputs:  numInputs,
		numOutputs: numOutputs,
	}
}

func (n *Network) feed(inputs []float64) {
	if len(inputs) != n.numInputs-1 {
		panic(`invalid input length`)
	}

	/* Set values for input layer */
	for idx, input := range inputs {
		node := n.nodes[idx].(*ConstantNode)
		node.value = input
	}

	/* Iterate over remaining nodes and update node values */
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

	for _, node := range dstNode.inputs {
		if node == srcNode {
			/* Edge already exists */
			return
		}
	}

	dstNode.inputs = append(dstNode.inputs, srcNode)
}

func (n *Network) changeRandomNodeType() {
	nodeIdx := rand.Intn(len(n.nodes)-n.numInputs) + n.numInputs
	node := n.nodes[nodeIdx].(*SumNode)
	node.op = RandomOperation()
}

func (n *Network) removeRandomEdge() {
	dstIdx := rand.Intn(len(n.nodes)-n.numInputs) + n.numInputs
	dstNode := n.nodes[dstIdx].(*SumNode)

	srcIdx := rand.Intn(len(dstNode.inputs))

	newInputs := dstNode.inputs[:srcIdx]
	newInputs = append(newInputs, dstNode.inputs[srcIdx+1:]...)
}

/* Splits a random edge between two nodes. */
func (n *Network) splitRandomEdge() {
	dstIdx := rand.Intn(len(n.nodes)-n.numInputs) + n.numInputs
	dstNode := n.nodes[dstIdx].(*SumNode)

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

	/* Create new middle node */
	middleNode := NewSumNode([]Node{srcNode})

	/* Replace src with middle node in dst's input list */
	newInputs := []Node{}
	for _, input := range dstNode.inputs {
		if input != srcNode {
			newInputs = append(newInputs, input)
		}
	}
	newInputs = append(newInputs, middleNode)
	dstNode.inputs = newInputs

	middleIdx := srcIdx + 1
	if srcIdx < n.numInputs {
		middleIdx = n.numInputs
	}
	n.nodes = append(n.nodes, nil)
	copy(n.nodes[middleIdx+1:], n.nodes[middleIdx:])
	n.nodes[middleIdx] = middleNode
}

func (n *Network) dedupEdges() {
	for _, node := range n.nodes[n.numInputs:] {
		node := node.(*SumNode)
		inputs := make(map[Node]bool)
		for _, input := range node.inputs {
			inputs[input] = true
		}
		newInputs := []Node{}
		for input, _ := range inputs {
			newInputs = append(newInputs, input)
		}
		node.inputs = newInputs
	}
}

func (n *Network) removeDeadEnds() {
	/* Dead ends are nodes that are not outputs or inputs and which are not inputs to any other node */

	changed := true
	for changed {
		changed = false

		usedAsInput := make(map[Node]bool)
		for _, node := range n.nodes[n.numInputs:] {
			node := node.(*SumNode)
			for _, input := range node.inputs {
				usedAsInput[input] = true
			}
		}

		removeIndices := []int{}
		for idx, node := range n.nodes[n.numInputs : len(n.nodes)-n.numOutputs] {
			idx += n.numInputs
			if !usedAsInput[node] {
				removeIndices = append(removeIndices, idx)
			}
		}

		for idx := len(removeIndices) - 1; idx >= 0; idx-- {
			n.nodes = append(n.nodes[:removeIndices[idx]], n.nodes[removeIndices[idx]+1:]...)
		}

		if len(removeIndices) != 0 {
			changed = true
		}
	}
}

func (n *Network) getOutput() []float64 {
	res := []float64{}

	for _, node := range n.nodes[len(n.nodes)-n.numOutputs:] {
		res = append(res, node.getValue())
	}

	return res
}

func (n *Network) Clone() *Network {
	clones := make(map[Node]Node)

	for _, node := range n.nodes {
		clones[node] = node.clone()
	}

	newNodes := []Node{}
	for _, node := range n.nodes {
		clone := clones[node]

		if sn, ok := clone.(*SumNode); ok {
			/* Need to update inputs */
			inputs := []Node{}
			for _, input := range node.(*SumNode).inputs {
				inputs = append(inputs, clones[input])
			}
			sn.inputs = inputs
		}

		newNodes = append(newNodes, clone)
	}

	return &Network{
		nodes:      newNodes,
		numInputs:  n.numInputs,
		numOutputs: n.numOutputs,
	}
}

func (n *Network) dumpDot(fname string, samples []Sample) {
	fh, err := os.Create(fname)
	if err != nil {
		log.Fatalln(`can't create file:`, err)
	}
	defer fh.Close()

	edgeCount := 0
	for _, node := range n.nodes[n.numInputs:] {
		node := node.(*SumNode)
		edgeCount += len(node.inputs)
	}

	fh.WriteString("digraph {\n")
	fh.WriteString("{ rank=same;\n")

	for idx, node := range n.nodes[:n.numInputs] {
		cn := node.(*ConstantNode)
		label := fmt.Sprintf("%f", cn.value)
		if idx == n.numInputs-1 {
			label = "BIAS"
		}
		fh.WriteString(fmt.Sprintf("\t\"%p\" [fontname=\"Ubuntu Mono\", style=filled, fillcolor=green, label=\"%d: %s\"];\n", cn, idx, label))
	}
	fh.WriteString("}\n")

	for idx, node := range n.nodes[n.numInputs:] {
		idx += n.numInputs
		node := node.(*SumNode)
		fh.WriteString("\t" + node.graphViz())
		for _, input := range node.inputs {
			fh.WriteString(fmt.Sprintf("\t\"%p\" -> \"%p\";\n", input, node))
		}
	}

	fh.WriteString("{ rank=same;\n")
	for _, node := range n.nodes[len(n.nodes)-n.numOutputs:] {
		fh.WriteString(fmt.Sprintf("\t\"%p\" [shape=doubleoctagon, style=\"rounded\"];\n", node))
	}
	fh.WriteString("}\n")

	fh.WriteString(fmt.Sprintf("\tlabel=\"Perf: %f, Edges: %d, Total Error: %f\";\n", n.performance(), edgeCount, n.totalError))
	fh.WriteString("\tlabelloc=top;\n")

	evalData := ""

	for _, sample := range samples {
		n.feed(sample.inputs)
		inputs := ""
		for _, val := range sample.inputs {
			inputs += fmt.Sprintf("% 3.0f", val)
		}
		outputs := ""
		for _, val := range n.getOutput() {
			outputs += fmt.Sprintf("% 3.2f", val)
		}
		targets := ""
		for _, val := range sample.targets {
			targets += fmt.Sprintf("% 3.2f", val)
		}

		evalData += fmt.Sprintf("[%s] -> [%s] [%s]\\l", inputs, outputs, targets)
	}

	fh.WriteString(fmt.Sprintf("eval [fontname=\"Ubuntu Mono\", label=\"%s\", shape=box]\n", evalData))

	fh.WriteString("}\n")
}

func (n *Network) structuralHash() string {
	h := fmt.Sprintf("%d|%d|", n.numInputs, n.numOutputs)

	nodeIdx := make(map[Node]int)

	for idx, node := range n.nodes {
		nodeIdx[node] = idx
		if node, ok := node.(*SumNode); ok {
			h += fmt.Sprintf("#%d|", node.op.Id())
			for _, input := range node.inputs {
				h += fmt.Sprintf("%d|", nodeIdx[input])
			}
			h += "#"
		}
	}

	return h
}

type Sample struct {
	inputs  []float64
	targets []float64
}

func (n *Network) sampleError(s Sample) float64 {
	delta := float64(0)

	n.feed(s.inputs)
	outputs := n.getOutput()

	for idx, output := range outputs {
		delta += math.Pow(output-s.targets[idx], 2)
	}

	return delta
}

func (n *Network) updateTotalError(samples []Sample) {
	totalError := float64(0)

	for _, s := range samples {
		totalError += n.sampleError(s)
	}

	n.totalError = totalError
}

const cutoff = 0.125

func (n *Network) performance() float64 {
	/* Count number of edges, discount total error for networks with low edge count */
	numEdges := 0
	for _, node := range n.nodes[n.numInputs:] {
		node := node.(*SumNode)
		numEdges += len(node.inputs)
	}

	dist := math.Pow((1-cutoff)*float64(n.totalError), 2) + math.Pow(cutoff*float64(numEdges), 2)

	return 1 / (dist + 1)
}
