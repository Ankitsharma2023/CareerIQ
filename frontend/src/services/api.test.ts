/**
 * The hand-rolled SSE buffer in analyzeResumeStream.
 *
 * This parser had no tests at all, and it is the single point every analysis
 * passes through: the whole report arrives as one `complete` event, so a bug
 * here loses the result after the server has done all the work. The buffer
 * exists because a TCP read boundary falls wherever it likes — the interesting
 * cases are all about an event arriving in pieces.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { analyzeResumeStream } from './api'

function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder()
  let index = 0
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: async () =>
          index < chunks.length
            ? { done: false, value: encoder.encode(chunks[index++]) }
            : { done: true, value: undefined },
        cancel: async () => {},
      }),
    },
  } as unknown as Response
}

const file = new File(['resume'], 'resume.pdf', { type: 'application/pdf' })

const progress = (n: number, m = 'working') =>
  `data: ${JSON.stringify({ event: 'progress', progress: n, message: m })}\n\n`
const complete = (result: unknown) =>
  `data: ${JSON.stringify({ event: 'complete', progress: 100, result })}\n\n`

describe('analyzeResumeStream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  async function run(chunks: string[]) {
    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(streamOf(chunks))
    const onProgress = vi.fn()
    const onComplete = vi.fn()
    const onError = vi.fn()
    const { done } = analyzeResumeStream(file, 'a job description', null, {
      onProgress,
      onComplete,
      onError,
    })
    await done
    return { onProgress, onComplete, onError }
  }

  it('delivers a whole event that arrives in one chunk', async () => {
    const { onProgress } = await run([progress(25, 'parsed')])
    expect(onProgress).toHaveBeenCalledWith(25, 'parsed')
  })

  it('delivers several events that arrive in one chunk', async () => {
    const { onProgress } = await run([progress(25) + progress(50) + progress(75)])
    expect(onProgress).toHaveBeenCalledTimes(3)
  })

  it('reassembles an event split across two chunks', async () => {
    // The reason the buffer exists. A read boundary can fall mid-JSON.
    const whole = complete({ semantic_match_score: 71 })
    const { onComplete } = await run([whole.slice(0, 20), whole.slice(20)])
    expect(onComplete).toHaveBeenCalledWith({ semantic_match_score: 71 })
  })

  it('reassembles an event split across many chunks', async () => {
    const whole = complete({ fit: { overall: 64 } })
    // Sliced by hand: /.{1,7}/g drops newlines, which would silently remove
    // the frame terminator and make this test pass for the wrong reason.
    const chunks: string[] = []
    for (let i = 0; i < whole.length; i += 7) chunks.push(whole.slice(i, i + 7))
    const { onComplete } = await run(chunks)
    expect(onComplete).toHaveBeenCalledWith({ fit: { overall: 64 } })
  })

  it('splits on the blank line, not on every newline', async () => {
    // A JSON payload can contain \n inside a string. Splitting on single
    // newlines would cut the payload in half and lose the whole report.
    const { onComplete } = await run([complete({ feedback: 'line one\nline two' })])
    expect(onComplete).toHaveBeenCalledWith({ feedback: 'line one\nline two' })
  })

  it('holds a partial trailing event rather than parsing it', async () => {
    const { onProgress, onError } = await run([progress(25) + 'data: {"event":"comp'])
    expect(onProgress).toHaveBeenCalledTimes(1)
    expect(onError).not.toHaveBeenCalled()
  })

  it('skips a malformed payload and keeps reading', async () => {
    // One bad frame must not cost the report that follows it.
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { onComplete } = await run([
      'data: {not json at all}\n\n' + complete({ ok: true }),
    ])
    expect(onComplete).toHaveBeenCalledWith({ ok: true })
  })

  it('ignores lines that are not data frames', async () => {
    const { onProgress } = await run([': keep-alive\n\n' + progress(40)])
    expect(onProgress).toHaveBeenCalledTimes(1)
  })

  it('routes a server error event to onError', async () => {
    const { onError, onComplete } = await run([
      `data: ${JSON.stringify({ event: 'error', message: 'Resume is image-based' })}\n\n`,
    ])
    expect(onError).toHaveBeenCalledWith('Resume is image-based')
    expect(onComplete).not.toHaveBeenCalled()
  })

  it('surfaces the server detail on a non-2xx response', async () => {
    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: 'File exceeds 5 MB limit.' }),
    } as unknown as Response)
    const { done } = analyzeResumeStream(file, 'jd', null, {})
    await expect(done).rejects.toThrow('File exceeds 5 MB limit.')
  })

  it('falls back to a readable message when the error body is not JSON', async () => {
    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error('not json')
      },
    } as unknown as Response)
    const { done } = analyzeResumeStream(file, 'jd', null, {})
    await expect(done).rejects.toThrow('Analysis failed')
  })

  it('explains an unreachable server rather than leaking the fetch error', async () => {
    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new TypeError('Failed to fetch'))
    const { done } = analyzeResumeStream(file, 'jd', null, {})
    await expect(done).rejects.toThrow('Could not reach the server')
  })

  it('emits a final event the server did not terminate with a blank line', async () => {
    // Found by a test, not in production: the loop only parsed on "\n\n", so a
    // last frame arriving without its terminator sat in the buffer and the
    // whole report was lost after the server had done all the work.
    const whole = complete({ semantic_match_score: 88 })
    const { onComplete } = await run([whole.replace(/\n\n$/, '')])
    expect(onComplete).toHaveBeenCalledWith({ semantic_match_score: 88 })
  })

  it('does not emit a truncated final frame', async () => {
    const { onComplete, onError } = await run(['data: {"event":"complete","res'])
    expect(onComplete).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()
  })

  it('resolves quietly when the caller aborts', async () => {
    // An abort is the user changing their mind, not a failure to report.
    const abortError = new Error('aborted')
    abortError.name = 'AbortError'
    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(abortError)
    const { done, abort } = analyzeResumeStream(file, 'jd', null, {})
    abort()
    await expect(done).resolves.toBeUndefined()
  })
})
