package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"
)

func MaxIdx(values []float64) int {
	maxSeen := math.Inf(-1)
	maxIdx := 0

	for idx, val := range values {
		if val > maxSeen {
			maxSeen = val
			maxIdx = idx
		}
	}

	return maxIdx
}

func profTask() {
	logger := log.New(os.Stdout, `[PROF ] `, log.LstdFlags)
	proffd, err := os.Create("cpuprofile.pprof")
	if err != nil {
		logger.Fatalln(`can't create CPU profile:`, err)
	}

	pprof.StartCPUProfile(proffd)
	logger.Println(`started profiling`)
	defer func() {
		pprof.StopCPUProfile()
		logger.Println(`stopped profiling`)
	}()

	select {
	case <-time.After(120 * time.Second):
		logger.Println(`profile timer expired`)
	}
}

func main() {
	logger := log.New(os.Stdout, `[MAIN ] `, log.LstdFlags)

	rand.Seed(time.Now().Unix())

	networks := []*Network{}

	samples := ReadMnist(`train`)

	for idx := 0; idx < epochSlice*2; idx++ {
		networks = append(networks, NewNetwork(28*28, 10))
	}
	/*
		samples := []Sample{
			Sample{inputs: []float64{-1, -1}, targets: []float64{-1}},
			Sample{inputs: []float64{-1, 1}, targets: []float64{1}},
			Sample{inputs: []float64{1, -1}, targets: []float64{1}},
			Sample{inputs: []float64{1, 1}, targets: []float64{-1}},
		}
		for idx := 0; idx < epochSlice*2; idx++ {
			networks = append(networks, NewNetwork(2, 1))
		}
	*/

	logger.Println(`training data loaded, starting training`)

	go profTask()

	networks = trainNetworks(networks, samples)

	for idx, net := range networks {
		net.dumpDot(fmt.Sprintf(`graphs/%03d.dot`, idx), samples)
	}

	logger.Println(`output of best network:`)
	errors := 0
	for _, s := range samples {
		networks[0].feed(s.inputs)
		output := MaxIdx(networks[0].getOutput())
		target := MaxIdx(s.targets)
		if output != target {
			errors += 1
		}
		logger.Println(`out:`, output, `target:`, target)
	}
	logger.Println(errors, `errors out of`, len(samples), `tests ->`, float64(errors)/float64(len(samples)), `error rate`)
}
