package main

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Network struct {
	layers       []*mat.Dense
	averageError float64
}

/* Unbiased new network */
func NewNetwork(layerSizes []int) *Network {
	layers := []*mat.Dense{}

	for idx, numInputs := range layerSizes[:len(layerSizes)-1] {
		numOutputs := layerSizes[idx+1]
		layer := mat.NewDense(numOutputs, numInputs, nil)
		for row := 0; row < numOutputs; row++ {
			for col := 0; col < numInputs; col++ {
				layer.Set(row, col, rand.NormFloat64())
			}
		}
		layers = append(layers, layer)
	}

	return &Network{
		layers: layers,
	}
}

/* This is a leaky RELU */
func (n *Network) nonlinearity(x float64) float64 {
	if x < 0 {
		return -0.01 * math.Exp(-x)
	}

	return x
}

func (n *Network) feed(inputs []float64, deltas []*mat.Dense) []float64 {
	output := mat.NewVecDense(len(inputs), inputs[:])

	for lidx, layer := range n.layers {
		if deltas != nil {
			var tempLayer mat.Dense
			tempLayer.Add(layer, deltas[lidx])
			layer = &tempLayer
		}

		var res mat.VecDense
		res.MulVec(layer, output)
		output = &res

		/* Apply nonlinearity */
		rows, _ := output.Dims()
		for row := 0; row < rows; row++ {
			output.SetVec(row, n.nonlinearity(output.AtVec(row)))
		}
	}

	res := make([]float64, len(inputs))

	for idx := 0; idx < output.Len(); idx++ {
		res[idx] = output.At(idx, 0)
	}

	return res
}

func (n *Network) error(targets, outputs []float64) float64 {
	delta := float64(0)

	for idx, t := range targets {
		delta += math.Pow(t-outputs[idx], 2)
	}

	return delta
}

type Sample struct {
	inputs  []float64
	targets []float64
}

const numConcurrency = 2
const numInnerConcurrency = 10
const numCandidates = 10 /* Number of mutations to evaluate */

func (n *Network) train(samples []Sample) {
	type Candidate struct {
		deltas []*mat.Dense
		error  float64
	}
	candidates := []*Candidate{}

	randSrc := rand.New(rand.NewSource(time.Now().Unix()))

	/* Generate a gaussian delta matrix for each candidate and each layer */
	for c := 0; c < numCandidates; c++ {
		deltas := []*mat.Dense{}
		for _, layer := range n.layers {
			rows, cols := layer.Dims()
			delta := mat.NewDense(rows, cols, nil)
			for row := 0; row < rows; row++ {
				for col := 0; col < cols; col++ {
					delta.Set(row, col, randSrc.NormFloat64())
				}
			}
			deltas = append(deltas, delta)
		}
		candidates = append(candidates, &Candidate{
			deltas: deltas,
		})
	}

	/* Evaluate all candidates */
	/* For all Candidates:
	       For all samples:
			     Gather sample error
				  Send up
			 Gather all sample errors
			 Update candidate.error
	*/

	var candidateWg sync.WaitGroup
	candidateChan := make(chan *Candidate, numConcurrency)

	for idx := 0; idx < numConcurrency; idx++ {
		go func() {
			for candidate := range candidateChan {
				var sampleWg sync.WaitGroup
				sampleChan := make(chan *Sample, numInnerConcurrency)
				errorChan := make(chan float64, numInnerConcurrency)

				for sIdx := 0; sIdx < numInnerConcurrency; sIdx++ {
					go func() {
						for sample := range sampleChan {
							output := n.feed(sample.inputs, candidate.deltas)
							errorChan <- n.error(sample.targets, output)
							sampleWg.Done()
						}
					}()
				}

				go func() {
					for _, sample := range samples {
						sampleWg.Add(1)
						sampleChan <- &sample
					}
					sampleWg.Wait()
					close(sampleChan)
					close(errorChan)
				}()

				sampleError := float64(0)
				for e := range errorChan {
					sampleError += e
				}
				candidate.error = sampleError / float64(len(samples))
				candidateWg.Done()
			}
		}()
	}

	for _, c := range candidates {
		candidateWg.Add(1)
		candidateChan <- c
	}

	candidateWg.Wait()
	close(candidateChan)

	/* Sort candidates by lowest error first */
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].error < candidates[j].error
	})

	/* Update network with candidate that produced the lowest error */
	for idx, delta := range candidates[0].deltas {
		n.layers[idx].Add(n.layers[idx], delta)
	}
}

func (n *Network) sampleError(s Sample) float64 {
	outputs := n.feed(s.inputs, nil)
	return n.error(s.targets, outputs)
}

func (n *Network) updateAverageError(samples []Sample) {
	averageError := float64(0)

	for _, s := range samples {
		averageError += n.sampleError(s)
	}

	n.averageError = averageError / float64(len(samples)+1)
}
