"""
Generate 100 sample prompts for batch evaluation of the MIT Course Catalog Chatbot.
Covers distribution, prereqs, scheduling, comparison, program-specific, exact course,
general advice, constraint bundles, and edge cases.
"""


def get_100_prompts() -> list[str]:
    """Return 100 diverse student-style prompts."""
    return [
        # ─── Distribution (CI-H, REST, HASS, LAB) ───
        "I need a CI-H this semester.",
        "What CI-H courses are about ethics or technology?",
        "I need a CI-H about AI or tech ethics.",
        "Which courses satisfy REST?",
        "I'm looking for REST courses that work with a CS background.",
        "What HASS-A courses are good for engineers?",
        "Do you have any CI-H with an afternoon section?",
        "I need to fulfill LAB. What are my options?",
        "REST options that don't require a lot of programming?",
        "CI-H that's writing-intensive?",
        "Any CI-H in philosophy or political science?",
        "REST course that fits a Tuesday/Thursday schedule?",
        "I need both CI-H and HASS. What do you recommend?",
        # ─── Prerequisites ───
        "What are the prerequisites for 6.3900?",
        "I've taken 18.06 and 6.100A. What can I take next in ML?",
        "What do I need before 6.1210?",
        "I don't have 6.031. What 6-3 courses can I still take?",
        "Prereqs for 24.131?",
        "Can I take 6.046 without 6.006?",
        "What's the prerequisite chain for 6.864?",
        # ─── Scheduling ───
        "I need a CI-H in the afternoon.",
        "Only afternoons work for me. What CI-H options are there?",
        "What courses are offered in Spring only?",
        "I'm free only TR 10–12. What fits?",
        "Any good courses that meet once a week?",
        "What's offered in Fall that has no Friday class?",
        "I need something that doesn't conflict with 18.06.",
        "Afternoon-only options for Course 6?",
        # ─── Comparison ───
        "Compare 6.3900 and 6.4100. Which should I take first?",
        "What's the difference between 6.1200 and 6.1210?",
        "18.06 vs 18.700 for linear algebra?",
        "Should I take 6.036 or 6.3900 for ML?",
        "Compare 24.131 and 17.01 for ethics.",
        "6.006 vs 6.046 for algorithms?",
        # ─── Program / degree ───
        "I'm a 6-3 junior. What electives do you recommend?",
        "I'm Course 8. I need a CI-H and prefer afternoon.",
        "For 6-3 graduation I still need 6.1200 and 6.1210. What order?",
        "I'm EECS. What HASS courses fit a technical schedule?",
        "6-1 senior. What's a good REST?",
        "I'm a TPP student. What CRE should I take?",
        "Course 2 major looking for a CI-H.",
        "I'm Course 17. Need a writing-intensive elective.",
        # ─── Exact course ───
        "Tell me about 24.131.",
        "What is 18.06?",
        "Info on 6.1210.",
        "Describe 17.01.",
        "What's 6.3900 about?",
        "Tell me about 4.031.",
        "What does 6.036 cover?",
        "Info on 21W.775.",
        # ─── General advice ───
        "I'm interested in AI. What should I take?",
        "Good courses for someone interested in systems?",
        "What are the best ML courses at MIT?",
        "I want to do robotics. What's the path?",
        "Recommend something for computational biology.",
        "I like theory. What 6-3 theory courses are there?",
        "What's a good first course in economics?",
        "I'm interested in climate and sustainability. Course suggestions?",
        "Recommend a seminar for freshmen.",
        # ─── Constraint bundles ───
        "I need a CI-H this semester, only afternoons, and I'm Course 8.",
        "6-3 junior, need CI-H, afternoon only, no 6.031.",
        "I need REST and I'm only free TR afternoon.",
        "CI-H that's in the afternoon and has no prereqs.",
        "I'm Course 6. I need a CI-H and an afternoon slot.",
        "Graduate student needing a HASS elective. Prefer small class.",
        # ─── Underspecified / probe ───
        "I want something in AI.",
        "What about ethics?",
        "Recommend a course.",
        "I need a HASS.",
        "What's good for 6.031?",
        "Do you have anything on NLP?",
        "Something fun for next semester.",
        # ─── No results / relax ───
        "I need a REST course that's AI-related, TR-only, and has no programming.",
        "CI-H in the morning only that's also a seminar.",
        "6.UAT undergrad thesis course that's not 6.UAT.",
        # ─── Edge / advisor ───
        "I'm struggling with 6.006. Who can I talk to?",
        "Who's my advisor for Course 6?",
        "Where do I check if a course counts for 6-3?",
        "Are there 6.UAT restrictions for 6.1200?",
        "I'm confused about the Communication Requirement.",
        # ─── Fill to 100 ───
        "What 6.xxx courses are offered in Spring?",
        "Easy CI-H for a busy semester?",
        "Recommend 3 courses for a 6-3 sophomore.",
        "What's the workload like for 18.06?",
        "Any CI-H that satisfies both HASS-H and CI-H?",
        "Courses similar to 24.131?",
        "What should I take after 6.100A?",
        "Best course to learn Python?",
        "I need 9 units of HASS. What do you suggest?",
        "Compare 6.100A and 16.C20.",
        "What's a good pairing with 6.1210?",
        "Any evening CI-H options?",
        "Recommend something with no final exam.",
        "What 6-3 courses are most popular?",
        "I want to explore quantum computing. Where do I start?",
        "Any 6.xxx with a project component?",
        "What CI-H is good for pre-med?",
        "Recommend a course that meets MW only.",
        "What's the difference between 6.100A and 6.100B?",
        "I need a light class to balance 6.1210.",
    ]


if __name__ == "__main__":
    prompts = get_100_prompts()
    assert len(prompts) == 100
    for i, p in enumerate(prompts):
        print(f"{i+1:3}. {p[:70]}{'...' if len(p) > 70 else ''}")
