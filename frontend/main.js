const generateBtn = document.getElementById("generateBtn");
const youtubeUrlInput = document.getElementById("youtubeUrl");
const status = document.getElementById("status");
const video = document.getElementById("videoPlayer");
const spinner = document.getElementById("spinner");
const thumbnail = document.getElementById("thumbnail");
const title = document.getElementById("title");
const preview = document.getElementById("preview");

async function fetchYouTubeMetadata(url) {
	try {
		const res = await fetch(`https://noembed.com/embed?url=${url}`);
		const data = await res.json();
		if (data.error) throw new Error("Invalid YouTube URL");
		return {
			thumbnail: data.thumbnail_url,
			title: data.title,
		};
	} catch {
		return null;
	}
}

generateBtn.addEventListener("click", async () => {
	const url = youtubeUrlInput.value.trim();
	status.textContent = "";
	video.style.display = "none";
	preview.style.display = "none";

	generateBtn.disabled = true;
	spinner.style.display = "block";
	status.style.color = "#00ccff";
	status.textContent = "⏳ 자막 생성 중입니다. 수십 초 이상 소요될 수 있어요...";

	const metadata = await fetchYouTubeMetadata(url);
	if (metadata) {
		preview.style.display = "block";
		thumbnail.src = metadata.thumbnail;
		title.textContent = metadata.title;
	}

	try {
		const response = await fetch("http://localhost:8000/generate", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ youtube_url: url }),
		});

		const contentType = response.headers.get("content-type") || "";

		if (response.ok && contentType.includes("video/mp4")) {
			// ✅ 정상 처리된 mp4 영상
			const blob = await response.blob();
			const videoUrl = URL.createObjectURL(blob);

			video.src = videoUrl;
			video.style.display = "block";
			status.textContent = "✅ 자막 영상 생성 완료!";
			status.style.color = "#00ff88";
			alert("🎉 자막 영상이 생성되었습니다!");
		} else if (contentType.includes("application/json")) {
			// ✅ 오류 응답 처리
			const data = await response.json();
			throw new Error(data.error || "❌ 서버 처리 중 오류가 발생했습니다.");
		} else {
			throw new Error("❌ 알 수 없는 형식의 응답입니다.");
		}
	} catch (err) {
		console.error(err);
		status.textContent = err.message || "❌ 알 수 없는 오류가 발생했습니다.";
		status.style.color = "red";
	} finally {
		spinner.style.display = "none";
		generateBtn.disabled = false;
	}
});

// ✅ 새로운 버튼 추가
const processBtn = document.getElementById("processBtn");

processBtn.addEventListener("click", async () => {
	const url = youtubeUrlInput.value.trim();

	if (!url) {
		alert("유튜브 URL을 입력하세요.");
		return;
	}

	status.textContent = "";
	video.style.display = "none";
	preview.style.display = "none";

	generateBtn.disabled = true;
	processBtn.disabled = true;
	spinner.style.display = "block";
	status.style.color = "#00ccff";
	status.textContent = "⏳ 서버 처리 중입니다. 잠시만 기다려주세요...";

	try {
		const response = await fetch("http://localhost:8000/process", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ youtube_url: url }),
		});

		const contentType = response.headers.get("content-type") || "";

		if (response.ok && contentType.includes("video/mp4")) {
			const blob = await response.blob();
			const videoUrl = URL.createObjectURL(blob);

			video.src = videoUrl;
			video.style.display = "block";
			status.textContent = "✅ 서버에서 자막 영상 생성 완료!";
			status.style.color = "#00ff88";
			alert("🎉 서버 자막 영상이 생성되었습니다!");
		} else {
			throw new Error("❌ 서버에서 알 수 없는 형식의 응답입니다.");
		}
	} catch (error) {
		console.error(error);
		status.textContent = error.message || "❌ 알 수 없는 오류가 발생했습니다.";
		status.style.color = "red";
	} finally {
		spinner.style.display = "none";
		generateBtn.disabled = false;
		processBtn.disabled = false;
	}
});
