package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
)

type Node interface {
	getValue() float64
	updateValue()
	clone() Node
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

func (c *ConstantNode) clone() Node {
	return &ConstantNode{c.value}
}

type Operation interface {
	Apply(float64) float64
}

type Invert struct{}

func (i Invert) Apply(n float64) float64 {
	return -n
}

type Tanh struct{}

func (t Tanh) Apply(n float64) float64 {
	return math.Tanh(n)
}

type Relu struct{}

func (r Relu) Apply(n float64) float64 {
	if n < 0 {
		return 0.01 * n
	}
	return n
}

type Sine struct{}
func (s Sine) Apply(n float64) float64 {
	return math.Sin(n)
}

type SumNode struct {
	op     Operation
	inputs []Node
	value  float64
}

func NewSumNode(inputs []Node) *SumNode {
	var op Operation

	switch rand.Intn(4) {
	case 0:
		op = Invert{}
	case 1:
		op = Tanh{}
	case 2:
		op = Relu{}
	case 3:
		op = Sine{}
	}

	return &SumNode{
		op:     op,
		inputs: inputs,
	}
}

func (s *SumNode) clone() Node {
	return &SumNode{
		op: s.op,
	}
}

func (s *SumNode) graphViz() string {
	label := fmt.Sprintf("%f", s.value)
	style := ""

	if s.op == (Invert{}) {
		style = ", style=filled, fillcolor=red"
		label = "± " + label
	} else if s.op == (Tanh{}) {
		style = ", style=filled, fillcolor=gray"
		label = "∫ " + label
	} else if s.op == (Sine{}) {
		label = "∿ " + label
	} else if s.op == (Relu{}) {
		label = "⦧ " + label
	}

	res := fmt.Sprintf("\t\"%p\" [label=\"%s\"", s, label)
	res += style + "]\n"
	return res
}

func (s *SumNode) updateValue() {
	sum := float64(0)

	for _, input := range s.inputs {
		sum += input.getValue()
	}

	s.value = s.op.Apply(sum)
}

func (s *SumNode) getValue() float64 {
	return s.value
}

type Network struct {
	nodes      []Node
	numInputs  int
	numOutputs int
	totalError float64
}

func NewNetwork(numInputs, numOutputs int) *Network {
	nodes := []Node{}
	for idx := 0; idx < numInputs; idx++ {
		nodes = append(nodes, &ConstantNode{})
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

	for _, node := range dstNode.inputs {
		if node == srcNode {
			/* Edge already exists */
			return
		}
	}

	dstNode.inputs = append(dstNode.inputs, srcNode)
}

func (n *Network) removeRandomEdge() {
	dstIdx := rand.Intn(len(n.nodes) - n.numInputs) + n.numInputs
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
		nodes: newNodes,
		numInputs: n.numInputs,
		numOutputs: n.numOutputs,
	}
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
			fh.WriteString("\t" + sn.graphViz())
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

func (n *Network) performance() float64 {
	if rand.Intn(30) == 0 {
		return 1 / (n.totalError + 1)
	}

	/* Count number of edges, discount total error for networks with low edge count */
	numEdges := 0
	for _, node := range n.nodes[n.numInputs:] {
		node := node.(*SumNode)
		numEdges += len(node.inputs)
	}

	return 1.0 / ((n.totalError * float64(numEdges)) + 1)
}

const numEpochs = 1000
const epochSlice = 50 // Number of survivors for each epoch

func main() {
	rand.Seed(time.Now().Unix())
	log.Println(`here we go`)

	samples := []Sample{
		Sample{[]float64{-1, -1}, []float64{-1}},
		Sample{[]float64{-1, 1}, []float64{1}},
		Sample{[]float64{1, -1}, []float64{1}},
		Sample{[]float64{1, 1}, []float64{-1}},
	}

	networks := []*Network{}
	for idx := 0; idx < epochSlice * 2; idx++ {
		networks = append(networks, NewNetwork(2, 1))
	}

	for epoch := 0; epoch < numEpochs; epoch++ {
		for _, net := range networks {
			net.updateTotalError(samples)
		}

		sort.Slice(networks, func(i, j int) bool {
			return networks[i].performance() > networks[j].performance()
		})

		/* Cull non-survivors */
		networks = networks[:epochSlice]

		/* Clone each survivor and mutate the clone */
		newNetworks := []*Network{}
		for _, net := range networks {
			clone := net.Clone()
			switch rand.Intn(5) {
			case 0:
				clone.removeRandomEdge()
			case 1,2:
				clone.addRandomEdge()
			case 3,4:
				clone.splitRandomEdge()
			}
			newNetworks = append(newNetworks, clone)
		}
		networks = append(networks, newNetworks...)

		if epoch % 10 == 0 {
			log.Println(`epoch`, epoch, `best performance`, networks[0].performance())
		}
	}

	for _, net := range networks {
		net.updateTotalError(samples)
	}

	sort.Slice(networks, func(i, j int) bool {
		return networks[i].performance() > networks[j].performance()
	})

	for idx, net := range networks {
		log.Println(idx, `->`, net.performance())
	}

	networks[0].dumpDot()
	log.Println(`output of best network:`)
	for _, s := range samples {
		networks[0].feed(s.inputs)
		log.Println(`in:`, s.inputs, `out:`, networks[0].getOutput(), `target:`, s.targets)
	}
}
