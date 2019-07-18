package main

import (
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	network "github.com/farhaven/nn-go"
)

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

	samples := readMnist(`train`)
	logger.Println(`training data loaded, starting training`)

	config := []network.LayerConfiguration{
		network.LayerConfiguration{28 * 28, nil},
		network.LayerConfiguration{800, network.SigmoidActivation{}},
		network.LayerConfiguration{40, network.LeakyRELUActivation{Leak: 0.01, Cap: 1e10}},
		network.LayerConfiguration{10, network.LeakyRELUActivation{Leak: 0.01, Cap: 1e10}},
	}
	network, err := network.NewNetwork(config)
	if err != nil {
		log.Fatalln(`can't create network:`, err)
	}

	/*
	samples := []Sample{
		Sample{[]float64{0, 0}, []float64{0}},
		Sample{[]float64{0, 1}, []float64{1}},
		Sample{[]float64{1, 0}, []float64{1}},
		Sample{[]float64{1, 1}, []float64{0}},
	}
	network := NewNetwork([]int{2, 3, 1})
	*/

	go profTask()

	trainNetwork(network, samples)
}
