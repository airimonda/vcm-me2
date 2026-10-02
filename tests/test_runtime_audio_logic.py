"""Endpointing, ring buffer, wake detector, window prep -- on synthetic audio."""
import numpy as np
import pytest

from runtime import prep
from runtime.endpoint import FRAME, SR, Endpointer, frame_db, noise_floor_db
from runtime.wake import RingBuffer, WakeDetector

RNG = np.random.RandomState(0)


def noise(sec, level_db=-60.0):
    return (RNG.randn(int(sec * SR)) * 10 ** (level_db / 20)).astype(np.float32)


def tone(sec, level_db=-25.0, f=300.0):
    t = np.arange(int(sec * SR)) / SR
    return (np.sin(2 * np.pi * f * t) * 10 ** (level_db / 20) * 1.414).astype(np.float32)


def feed_all(ep, audio, block=1600):
    for i in range(0, len(audio), block):
        if ep.feed(audio[i:i + block]):
            break
    return ep


def utterance(lead=0.6, speech=1.0, tail=2.0):
    return np.concatenate([noise(lead), tone(speech) + noise(speech), noise(tail)])


# ---------------------------------------------------------------- endpointer
def test_stops_about_0p7s_after_speech_ends():
    a = utterance(lead=0.6, speech=1.0, tail=2.0)
    ep = feed_all(Endpointer(noise_db=-60), a)
    assert ep.done and ep.reason == "end_of_speech" and ep.has_speech
    speech_end = 0.6 + 1.0
    stop = ep.n_samples / SR
    assert speech_end + 0.6 <= stop <= speech_end + 0.95            # ~0.7 s hangover (+ one 0.1 s block)
    assert ep.speech_end_s == pytest.approx(speech_end, abs=0.05)
    assert len(ep.audio) == ep.n_samples                            # every sample is kept for the model


def test_respects_minimum_length():
    # a 0.2 s blip: silence criterion would fire at ~0.9 s but min_s = 1.2 s keeps listening until then
    a = np.concatenate([tone(0.3), noise(3.0)])
    ep = feed_all(Endpointer(noise_db=-60, min_s=1.2, start_ignore_s=0.0, min_speech_s=0.1), a)
    assert ep.reason == "end_of_speech" and ep.n_samples / SR >= 1.2


def test_pause_inside_the_command_does_not_end_it():
    a = np.concatenate([noise(0.3), tone(0.8), noise(0.4), tone(0.8), noise(2.0)])     # 0.4 s gap < 0.7 s
    ep = feed_all(Endpointer(noise_db=-60), a)
    assert ep.reason == "end_of_speech" and ep.n_samples / SR > 0.3 + 0.8 + 0.4 + 0.8 + 0.6


def test_no_speech_times_out():
    ep = feed_all(Endpointer(noise_db=-60, no_speech_timeout_s=2.0), noise(5.0))
    assert ep.done and ep.reason == "no_speech" and not ep.has_speech
    assert ep.n_samples / SR == pytest.approx(2.0, abs=0.2)


def test_hard_cap_at_max_s():
    ep = feed_all(Endpointer(noise_db=-60, max_s=5.0), tone(8.0))                  # never goes quiet
    assert ep.reason == "max_length" and ep.n_samples == 5 * SR


def test_start_ignore_window_ignores_a_chime_but_keeps_its_audio():
    a = np.concatenate([tone(0.15, f=1200.0), noise(3.5)])                         # a chime and nothing else
    ep = feed_all(Endpointer(noise_db=-60, start_ignore_s=0.25, no_speech_timeout_s=2.0), a)
    assert ep.reason == "no_speech"
    ep = Endpointer(noise_db=-60, start_ignore_s=0.0, no_speech_timeout_s=2.0, min_speech_s=0.1)
    feed_all(ep, a)
    assert ep.has_speech                                                           # without the guard it would count


def test_threshold_follows_the_noise_floor():
    loud_room = noise(6.0, level_db=-40.0)
    speech = np.concatenate([noise(0.5, -40), tone(1.0, -20) + noise(1.0, -40), noise(2.0, -40)])
    nf = noise_floor_db(loud_room)
    assert -43 < nf < -37
    ep = feed_all(Endpointer(noise_db=nf), speech)
    assert ep.reason == "end_of_speech"
    # with the default quiet-room floor the same recording never goes silent
    ep2 = feed_all(Endpointer(noise_db=-60, max_s=5.0), speech)
    assert not ep2.done                       # room noise sits above that threshold, so it never goes quiet


def test_arbitrary_chunk_sizes_give_the_same_result():
    a = utterance()
    r = []
    for blk in (160, 1600, 3333):
        ep = feed_all(Endpointer(noise_db=-60), a, block=blk)
        r.append((ep.reason, round(ep.speech_end_s, 2)))
    assert len(set(r)) == 1


def test_eof_closes_open_capture():
    ep = Endpointer(noise_db=-60)
    ep.feed(np.concatenate([noise(0.4), tone(0.5)]))
    ep.finish_eof()
    assert ep.done and ep.reason == "source_ended"


def test_frame_db_levels():
    assert frame_db(tone(0.1, level_db=-20))[2] == pytest.approx(-20, abs=1.0)
    assert len(frame_db(np.zeros(FRAME * 3 + 5))) == 3


# ---------------------------------------------------------------- ring buffer / wake detector
def test_ring_buffer_keeps_the_last_samples():
    r = RingBuffer(10)
    r.append(np.arange(4, dtype=np.float32))
    assert r.filled == 4 and list(r.last(10)) == [0, 1, 2, 3]
    r.append(np.arange(4, 12, dtype=np.float32))
    assert list(r.last(10)) == list(range(2, 12))
    r.append(np.arange(100, 130, dtype=np.float32))                  # bigger than capacity
    assert list(r.last(3)) == [127, 128, 129]


def make_detector(probs, **kw):
    seen = []

    def score(x):
        seen.append(len(x))
        return probs.pop(0) if probs else 0.0, 1.0
    return WakeDetector(score, window=24000, hop_s=0.25, threshold=0.55, cooldown_s=1.0, **kw), seen


def drive(det, seconds, block=1600):
    events = []
    for i in range(int(seconds * SR / block)):
        ev = det.feed(np.zeros(block, np.float32))
        if ev:
            events.append((i * block / SR, ev))
    return events


def test_wake_scores_every_hop_and_fires_at_threshold():
    det, seen = make_detector([0.1, 0.2, 0.9, 0.9])
    ev = drive(det, 2.0)
    assert len(ev) == 1 and ev[0][1].prob == pytest.approx(0.9)
    assert ev[0][0] == pytest.approx(0.8, abs=0.11)               # 3rd score = 0.75 s .. 0.8 s
    assert len(seen) >= 3 and all(n <= 24000 for n in seen)


def test_threshold_is_inclusive_and_below_does_not_fire():
    det, _ = make_detector([0.5499, 0.55])
    ev = drive(det, 1.0)
    assert len(ev) == 1 and ev[0][1].prob == 0.55


def test_consecutive_windows_required():
    det, _ = make_detector([0.9, 0.1, 0.9, 0.9], consecutive=2)
    assert len(drive(det, 2.0)) == 1                                 # only the 3rd+4th pair fires


def test_held_until_release_then_cooldown():
    det, _ = make_detector([0.9] * 50)
    assert len(drive(det, 1.0)) == 1                                 # fires once, then held (a turn is running)
    det.release()
    ev = drive(det, 2.0)
    assert ev and ev[0][0] >= 1.0 - 0.1                              # not before the 1 s cooldown


def test_wake_event_is_not_retriggered_by_the_command_audio():
    det, _ = make_detector([0.9] * 50)
    assert len(drive(det, 1.0)) == 1
    assert det.ring.filled > 0
    det.release()
    assert det.ring.filled == 0


# ---------------------------------------------------------------- window prep
def test_prepare_window_shape_range_and_centering():
    x = np.concatenate([noise(1.0), tone(1.0), noise(1.0)])
    w = prep.prepare_window(x)
    assert w.shape == (prep.WINDOW,) and w.dtype == np.float32 and np.abs(w).max() <= 1.0
    active = np.nonzero(np.abs(w) > 0.01)[0]
    assert abs((active[0] + active[-1]) / 2 - prep.WINDOW / 2) < 0.4 * SR      # speech sits in the middle


def test_prepare_window_long_audio_keeps_the_loudest_5s():
    x = np.concatenate([noise(2.0), tone(4.5), noise(2.0)])
    w = prep.prepare_window(x)
    assert w.shape == (prep.WINDOW,)
    assert np.abs(w).max() > 0.05
