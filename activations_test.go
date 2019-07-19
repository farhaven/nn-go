package network

import "testing"

func TestLeakyRELUActivationForward(t *testing.T) {
	act := LeakyRELUActivation{Leak: 0.001, Cap: 10}

	if a := act.Forward(0); a != 0 {
		t.Error(`0 -> `, a)
	}

	if a := act.Forward(1); a != 1 {
		t.Error(`1 -> `, a)
	}

	if a := act.Forward(-1); a != -0.001 {
		t.Error(`-1 -> `, a)
	}

	// Test cap
	if a := act.Forward(11); a != 10 {
		t.Error(`cap failed: `, a)
	}

	if a := act.Forward(-11.0 / 0.001); a != -10 {
		t.Error(`cap failed: `, a)
	}
}

func TestLeakyRELUActivationBackward(t *testing.T) {
	act := LeakyRELUActivation{Leak: 0.001, Cap: 1e10}

	if a := act.Backward(0); a != 0.5 {
		t.Error(`0 -> `, a)
	}

	if a := act.Backward(1); a != 1 {
		t.Error(`1 -> `, a)
	}

	if a := act.Backward(-1); a != 0.001 {
		t.Error(`-1 -> `, a)
	}
}
