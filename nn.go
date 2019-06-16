package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"
)

const numEpochs = 1000
const epochSlice = 50 // Number of survivors for each epoch

func main() {
	rand.Seed(time.Now().Unix())
	log.Println(`here we go`)

	samples := []Sample{}
	for idx := 0; idx < 30; idx++ {
		in := float64(idx) * ((math.Pi * 2) / 30)
		t1 := math.Cos(in)
		t2 := math.Sin(in)
		t3 := math.Tanh(in)

		samples = append(samples, Sample{[]float64{in}, []float64{t1, t2, t3}})
	}

	networks := []*Network{}
	for idx := 0; idx < epochSlice*2; idx++ {
		networks = append(networks, NewNetwork(1, 3))
	}

trainingLoop:
	for epoch := 0; epoch < numEpochs; epoch++ {
		for _, net := range networks {
			net.updateTotalError(samples)
		}

		sort.Slice(networks, func(i, j int) bool {
			return networks[i].performance() > networks[j].performance()
		})

		/* Cull non-survivors */
		networks = networks[:epochSlice]

		for len(networks) <= epochSlice {
			/* Clone each survivor and mutate the clone */
			newNetworks := []*Network{}
			for _, net := range networks {
				clone := net.Clone()
				clone.mutate()
				newNetworks = append(newNetworks, clone)
			}
			networks = append(networks, newNetworks...)

			if epoch%10 == 0 {
				bestPerf := networks[0].performance()
				log.Println(`epoch`, epoch, `best performance`, bestPerf)
				if bestPerf == 1 && epoch > 0 {
					break trainingLoop
				}
			}

			/* Cull duplicates */
			structures := make(map[string]*Network)
			for _, net := range networks {
				structures[net.structuralHash()] = net
			}
			networks = []*Network{}
			for _, net := range structures {
				networks = append(networks, net)
			}
		}

		for _, net := range networks {
			net.dedupEdges()
		}
	}

	for _, net := range networks {
		net.updateTotalError(samples)
		net.removeDeadEnds()
	}

	sort.Slice(networks, func(i, j int) bool {
		return networks[i].performance() > networks[j].performance()
	})

	for idx, net := range networks {
		net.dumpDot(fmt.Sprintf(`graphs/%03d.dot`, idx), samples)
	}

	log.Println(`output of best network:`)
	for _, s := range samples {
		networks[0].feed(s.inputs)
		log.Println(`in:`, s.inputs, `out:`, networks[0].getOutput(), `target:`, s.targets)
	}
}
