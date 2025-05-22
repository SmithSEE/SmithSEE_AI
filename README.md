# 1. 깃 레포지토리 clone

```
git clone https://github.com/SmithSEE/SmithSEE_AI.git
```

# 2. 최신 main 브랜치 가져오기(매번 처음에 실시)

```
git checkout main
git pull origin main
```

# 3. 개인 작업 브랜치 생성

```
git checkout -b "내 이름"
```

# 4. 코드 작업 후 커밋

```
git add .
git commit -m "커밋 내용 기입"
```

# 5. 브랜치에 push

```
git push origin "내 이름"
```

# 6. PR 생성

![alt text](/Image//PR_1.png)

- Pull requset > New pull request

![alt text](/Image/PR_2.png)

- compare 브랜치 "내 브랜치" 선택

![alt text](/Image/PR_3.png)

- Create pull request

## PR 컨벤션

### 제목

    Feat: "로그인 기능 구현"

- 제목과 본문을 한 줄 띄우고 콜론(:)으로 분리
- 제목은 명령어, 개조식으로 작성

| 제목 태그 이름 | 설명                                                |
| -------------- | --------------------------------------------------- |
| Feat           | 새로운 기능을 추가할 경우                           |
| Fix            | 버그를 고친 경우                                    |
| Design         | CSS 등 사용자 UI 디자인 변경                        |
| Test           | 테스트 추가, 테스트 리팩토링 (프로덕션 코드 변경 X) |
| Rename         | 파일 혹은 폴더명을 수정하거나 옮기는 작업만인 경우  |

### 본문

- 선택사항이기 때문에 모든 커밋에 본문을 작성할 필요는 없다
- 부연설명이 필요하거나 커밋의 이유를 설명할 경우 작성
- 본문은 어떻게 변경했는지 보다 무엇을 변경했는지, 왜 변경했는지 에 맞추어 작성

![alt text](/Image/PR_4.png)

- 컨벤션 잘 작성 후, Create pull request

![alt text](/Image/PR_5.png)

- 코드 리뷰 후, Merge pull request
- Confirm Commit
