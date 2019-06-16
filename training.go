package main

import (
	"log"
	"sort"
)

func trainNetworks(networks []*Network, samples []Sample) []*Network {
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

	return networks
}
