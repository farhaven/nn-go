package network

import (
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"
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

	config := []LayerConfiguration{
		LayerConfiguration{28 * 28, nil},
		LayerConfiguration{800, SigmoidActivation{}},
		LayerConfiguration{40, SigmoidActivation{}},
		LayerConfiguration{10, SigmoidActivation{}},
	}
	network, err := NewNetwork(config)
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
